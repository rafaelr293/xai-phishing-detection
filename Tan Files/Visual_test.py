import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.multiprocessing
# Windows' default DataLoader worker→main transport uses file-mapping-based
# shared memory, which is what ran out with "Couldn't open shared file
# mapping ... error code: 1455" (ERROR_COMMITMENT_LIMIT — virtual memory /
# page file exhausted). 'file_descriptor'-style sharing isn't available on
# Windows, but keeping worker count low + not holding workers persistently
# alive (see NUM_WORKERS / persistent_workers below) is what actually fixes
# this on Windows; this call is a no-op there but harmless, and matters if
# this script is ever run on Linux instead.
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_curve, auc)

# ─────────────────────────────────────────
# FUNCTION / CLASS DEFINITIONS
# (kept at module level — Windows DataLoader workers need to import
# these when they spawn, so they must NOT be inside the __main__ guard)
# ─────────────────────────────────────────

def load_phish360(data_dir=None):
    """
    Load the Phish360 dataset — confirmed structure:
      Phish360_legit.parquet / Phish360_phish.parquet, each with
      columns: sample_id, URL, full_html, Class, image_path, raw_label
      'image_path' is a path RELATIVE to data_dir (e.g. 'screenshots/L00003_legitimate.png').

    data_dir defaults to the folder this script itself is saved in —
    move this script into your phish360-data folder (alongside the
    two .parquet files and the images/screenshots folders) and it
    will find everything automatically, regardless of the terminal's
    current working directory.
    """
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))

    frames = []
    for fname, label_val in [('Phish360_legit.parquet', 0), ('Phish360_phish.parquet', 1)]:
        full_path = os.path.join(data_dir, fname)
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Could not find '{fname}' in {data_dir}. "
                f"Either move this script into that folder, or pass "
                f"data_dir explicitly to load_phish360()."
            )
        part = pd.read_parquet(full_path)
        part['label_num'] = label_val  # legit=0 / phish=1, known from which file it came from
        frames.append(part)
        print(f"  Loaded {fname} → {len(part):,} samples")

    df = pd.concat(frames, ignore_index=True)

    if 'image_path' not in df.columns:
        raise KeyError(
            f"Expected an 'image_path' column. Found columns: {list(df.columns)}"
        )

    # Build the full, absolute path to each screenshot file
    df['full_image_path'] = df['image_path'].apply(lambda p: os.path.join(data_dir, p))

    # NOTE: 'full_html' column is available too (raw page HTML per sample) —
    # not used by this CNN-only module, but worth keeping in mind if the
    # visual module later gets extended to combine HTML + image features.

    print(f"\n  Total samples → {len(df):,}")
    return df


def _check_one_image(args):
    """
    Worker function for scan_corrupted_images — must be a plain
    module-level function (not a closure/lambda) so it can be pickled
    and sent to child processes on Windows (which uses spawn, not fork).
    Takes (idx, path), returns (idx, path, error_string_or_None).
    """
    idx, path = args
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"file not found: {path}")
        # verify() does a cheap structural check without decoding
        # the full image — good first-pass filter
        with Image.open(path) as img:
            img.verify()
        # verify() leaves the file handle unusable afterwards, so
        # re-open fresh and actually decode + convert — this is what
        # catches the P19804_apple.png-style failures that verify()
        # alone can miss
        with Image.open(path) as img:
            img.convert('RGB')
        return (idx, path, None)
    except Exception as e:
        return (idx, path, f"{type(e).__name__}: {e}")


def scan_corrupted_images(df, path_col='full_image_path', verbose=True, num_workers=8):
    """
    Opens every image referenced in df[path_col] to check it's actually
    readable (catches missing files, 0-byte files, truncated downloads,
    HTML-saved-as-.png, wrong extensions, etc.) — the exact failure mode
    that otherwise crashes training mid-epoch with
    PIL.UnidentifiedImageError inside a DataLoader worker.

    Runs inline before training starts, so bad samples are dropped up
    front instead of relying on a separate scan script + saved indices
    file that can drift out of sync with the current dataset.

    Parallelized across `num_workers` processes (default 8) — opening/
    decoding each image is CPU-bound and fully independent per file, so
    this scales close to linearly with core count and turns what could
    be a slow multi-minute serial pass over ~10k+ images into a much
    shorter one.

    Returns a list of dataframe indices that failed to open.
    """
    import concurrent.futures

    total = len(df)
    tasks = list(zip(df.index, df[path_col]))
    bad_indices = []
    done = 0

    if verbose:
        print(f"\nScanning {total:,} images for corruption using {num_workers} processes...")

    # chunksize > 1 cuts down on inter-process communication overhead
    # when there are many small, fast tasks like this
    chunksize = max(1, total // (num_workers * 20))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, path, error in executor.map(_check_one_image, tasks, chunksize=chunksize):
            done += 1
            if error is not None:
                bad_indices.append(idx)
                if verbose:
                    print(f"  ✗ [{idx}] {path}  →  {error}")

            if verbose and done % 2000 == 0:
                print(f"  ...scanned {done:,}/{total:,}")

    if verbose:
        pct = (len(bad_indices) / total * 100) if total else 0.0
        print(f"\nScan complete → {len(bad_indices)} corrupted/unreadable image(s) "
              f"out of {total:,} ({pct:.2f}%)")

    return bad_indices


IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.2),   # mild augmentation — login pages are rarely mirrored, keep this light
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class Phish360Dataset(Dataset):
    """
    Wraps the loaded dataframe. Opens each screenshot directly from
    'full_image_path' (already resolved to an absolute path in
    load_phish360()).
    """
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['full_image_path']

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Row {idx}: image not found at {path}. "
                f"Check that images/screenshots weren't moved after loading."
            )

        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row['label_num'], dtype=torch.long)
        return img, label


def build_model(device):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    # Replace the final classifier layer for binary output (legit vs phishing)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    return model.to(device)


def run_epoch(model, loader, device, criterion, optimizer=None, train_mode=True):
    model.train() if train_mode else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train_mode):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train_mode:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# This is the computer-vision equivalent of the SHAP explanations used
# in the email/URL modules — it highlights which region of the
# screenshot the model focused on (e.g. a fake login form, a logo).
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=1):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + 1e-8)
        return heatmap


def plot_confusion_matrix(cm_data, title, filename):
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'],
                annot_kws={"size": 14})
    plt.title(title, fontsize=13)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved → {filename}")


def plot_roc(y_true, y_scores, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved → {filename}")


# ─────────────────────────────────────────
# EVERYTHING BELOW ACTUALLY RUNS THE SCRIPT.
# Wrapped in `if __name__ == "__main__":` — REQUIRED on Windows once
# num_workers > 0, otherwise each DataLoader worker re-imports this
# file and re-spawns its own workers infinitely (RuntimeError/hang).
# ─────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("   PHISHING WEBPAGE CLASSIFIER — EfficientNet-B0")
    print("=" * 55)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {DEVICE}")

    # Let PyTorch use multiple CPU cores for its own tensor math
    # (separate from DataLoader's num_workers, which parallelises
    # the image loading/preprocessing step instead).
    torch.set_num_threads(8)

    print("\nLoading dataset...")
    df = load_phish360(data_dir='phish360-data')

    # ─────────────────────────────────────────
    # SCAN FOR CORRUPTED IMAGES (runs every time, before training)
    # Replaces the old workflow of running scan_phish360_images.py
    # separately and reading back phish360_bad_indices.txt — that file
    # could go stale if the dataset changed. Scanning inline guarantees
    # the exclusion list always matches the data actually being loaded.
    # ─────────────────────────────────────────
    bad_indices = scan_corrupted_images(df, path_col='full_image_path')

    if bad_indices:
        print(f"Excluding {len(bad_indices)} corrupted sample(s) from training")
        df = df.drop(index=bad_indices).reset_index(drop=True)
    else:
        print("No corrupted images found.")

    phishing_count = df['label_num'].sum()
    legit_count = len(df) - phishing_count
    print(f"\n  Total samples      : {len(df):,}")
    print(f"  Phishing samples   : {phishing_count:,}  ({phishing_count/len(df)*100:.1f}%)")
    print(f"  Legitimate samples : {legit_count:,}  ({legit_count/len(df)*100:.1f}%)")

    # Drop heavy columns (full_html, URL text, etc.) not needed by this
    # CV-only model. Keeping them would mean every DataLoader worker
    # process (Windows uses pickling to send data to workers) has to
    # receive a copy of the entire raw HTML text for every sample —
    # easily exceeding 2GB total and triggering an OSError on Windows.
    df = df[['full_image_path', 'label_num']].copy()

    # ─────────────────────────────────────────
    # TRAIN / VAL / TEST SPLIT
    # ─────────────────────────────────────────
    full_dataset = Phish360Dataset(df, transform=None)  # transform applied per-split below

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply the right transform per split (train gets augmentation, val/test don't)
    train_subset.dataset = Phish360Dataset(df, transform=train_transform)
    val_subset.dataset = Phish360Dataset(df, transform=eval_transform)
    test_subset.dataset = Phish360Dataset(df, transform=eval_transform)

    BATCH_SIZE = 32
    # NUM_WORKERS=2 — confirmed root cause of the epoch-7 crash was
    # "RuntimeError: Couldn't open shared file mapping ... error code: 1455"
    # (ERROR_COMMITMENT_LIMIT), i.e. Windows virtual memory/page file
    # exhaustion from DataLoader worker↔main shared-memory transport
    # accumulating over many epochs. Lower worker count = fewer simultaneous
    # shared-memory allocations. If this still happens at 2, drop to 0
    # (fully single-process loading — slower but immune to this failure
    # mode) and/or increase the Windows page file size (see note below).
    NUM_WORKERS = 2

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0))
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0))
    # test_loader is only used AFTER the training loop finishes, so it has no
    # reason to hold worker processes alive (persistent_workers=True) for the
    # entire duration of training — that was pure wasted RAM/shared-memory
    # handles. Built with num_workers=0 instead, only used at the very end.
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    print(f"\nTraining samples : {train_size:,}")
    print(f"Validation samples: {val_size:,}")
    print(f"Testing samples  : {test_size:,}")

    # ─────────────────────────────────────────
    # MODEL
    # ─────────────────────────────────────────
    model = build_model(DEVICE)

    # NOTE: full fine-tuning (all layers trainable), matching the approach
    # used in the earlier deepfake-detection project. If training is too
    # slow or overfits on Phish360's smaller sample size, an easy adjustment
    # is to freeze model.features and only train model.classifier.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    # ─────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────
    NUM_EPOCHS = 3  # lowered to 3 for now — just need a functional run + screenshots
                    # (confusion matrix, ROC, Grad-CAM) to demonstrate the pipeline works;
                    # crashed at epoch 7/10 before anyway due to Windows shared-memory
                    # exhaustion, so this also sidesteps that until the page-file fix
                    # is confirmed. Raise back to ~10+ once accuracy actually matters.

    # Optional: log free RAM each epoch, if psutil is available. This is
    # the main diagnostic for the "process just vanishes, no traceback"
    # symptom — on Windows, running out of RAM with several DataLoader
    # worker processes alive often kills python without a clean Python
    # exception, so seeing free memory trending toward 0 right before a
    # silent death confirms OOM as the cause.
    try:
        import psutil
        _HAS_PSUTIL = True
    except ImportError:
        _HAS_PSUTIL = False
        print("(tip: `pip install psutil` to see free-RAM per epoch, "
              "useful for diagnosing silent crashes)")

    print("\nTraining model...")
    best_val_loss = float('inf')
    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc = run_epoch(model, train_loader, DEVICE, criterion, optimizer, train_mode=True)
            val_loss, val_acc = run_epoch(model, val_loader, DEVICE, criterion, train_mode=False)
            scheduler.step(val_loss)

            log_line = (f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
                        f"Train loss: {train_loss:.4f} acc: {train_acc*100:.2f}% | "
                        f"Val loss: {val_loss:.4f} acc: {val_acc*100:.2f}%")
            if _HAS_PSUTIL:
                mem = psutil.virtual_memory()
                log_line += f" | RAM free: {mem.available / 1e9:.2f} GB ({mem.percent:.0f}% used)"
            print(log_line)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'phish360_efficientnet_b0_best.pt')
                print(f"  → New best model saved (val loss {val_loss:.4f})")
    except Exception:
        # Without this, an exception raised inside a DataLoader worker
        # (or an OOM-related crash the OS DOES manage to hand back to
        # Python) can otherwise get lost/truncated depending on how the
        # terminal is being run. Printing the full traceback explicitly
        # here, plus flushing stdout, maximizes the chance you actually
        # see the real error instead of the process just going silent.
        import traceback
        print("\n" + "=" * 55)
        print("TRAINING CRASHED — full traceback below:")
        print("=" * 55)
        traceback.print_exc()
        import sys
        sys.stdout.flush()
        raise

    print("\nDone! Loading best checkpoint for evaluation...")
    model.load_state_dict(torch.load('phish360_efficientnet_b0_best.pt'))

    # ─────────────────────────────────────────
    # EVALUATE ON TEST SET
    # ─────────────────────────────────────────
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of "phishing"
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print("\n" + "=" * 55)
    print("RESULTS — Phish360 (test set)")
    print("=" * 55)
    print(f"Accuracy : {accuracy_score(all_labels, all_preds)*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Legitimate', 'Phishing']))

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue  Negatives (Legit → Legit)     : {tn}  ✅")
    print(f"True  Positives (Phish → Phish)     : {tp}  ✅")
    print(f"False Positives (Legit → Phish)     : {fp}  ⚠️")
    print(f"False Negatives (Phish → Legit)     : {fn}  🚨  (missed phishing — check this closely)")

    # ─────────────────────────────────────────
    # CHARTS
    # ─────────────────────────────────────────
    plot_confusion_matrix(cm, 'Confusion Matrix — Phish360', 'visual_confusion_matrix.png')
    plot_roc(all_labels, all_probs, 'ROC Curve — Phish360', 'visual_roc_curve.png')

    # ─────────────────────────────────────────
    # GRAD-CAM SETUP
    # ─────────────────────────────────────────
    # EfficientNet-B0's last conv block — good default target layer for Grad-CAM
    gradcam = GradCAM(model, model.features[-1])

    def explain_prediction(image_path, save_path='gradcam_output.png'):
        """
        Runs a prediction on a single image and saves a Grad-CAM overlay
        showing which region influenced the decision most.
        """
        img = Image.open(image_path).convert('RGB')
        input_tensor = eval_transform(img).unsqueeze(0).to(DEVICE)

        heatmap = gradcam.generate(input_tensor, class_idx=1)
        heatmap_resized = np.array(Image.fromarray(heatmap).resize((IMG_SIZE, IMG_SIZE)))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img.resize((IMG_SIZE, IMG_SIZE)))
        axes[0].set_title('Original Screenshot')
        axes[0].axis('off')

        axes[1].imshow(img.resize((IMG_SIZE, IMG_SIZE)))
        axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.5)
        axes[1].set_title('Grad-CAM — Model Focus Area')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"Saved → {save_path}")

    def predict_image(image_path, show_gradcam=True):
        img = Image.open(image_path).convert('RGB')
        input_tensor = eval_transform(img).unsqueeze(0).to(DEVICE)

        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        phish_p = probs[1].item() * 100
        legit_p = probs[0].item() * 100
        pred = 1 if phish_p > legit_p else 0
        label = "🚨 PHISHING" if pred == 1 else "✅ LEGITIMATE"

        # NOTE: placeholder thresholds — recalibrate later, same as the
        # other two modules.
        if probs[1].item() >= 0.85:   risk = "CRITICAL 🔴"
        elif probs[1].item() >= 0.60: risk = "HIGH     🟠"
        elif probs[1].item() >= 0.35: risk = "MEDIUM   🟡"
        else:                          risk = "LOW      🟢"

        print(f"\n{'─'*55}")
        print(f"Image  : {image_path}")
        print(f"Result : {label}")
        print(f"Legitimate probability : {legit_p:.2f}%")
        print(f"Phishing   probability : {phish_p:.2f}%")
        print(f"Risk Level              : {risk}")
        print(f"{'─'*55}")

        if show_gradcam:
            explain_prediction(image_path)

        return pred, probs.cpu().numpy()

    # ─────────────────────────────────────────
    # INTERACTIVE MODE
    # ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("     INTERACTIVE WEBPAGE SCREENSHOT TESTER")
    print("=" * 55)
    print("Commands:")
    print("  Enter a path to a local screenshot image (.png/.jpg) to test it")
    print("  'quit' → exit")
    print("=" * 55)

    while True:
        print()
        user_input = input("Image path (or 'quit'): ").strip().strip('"').strip("'")

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nFiles saved:")
            print("  → visual_confusion_matrix.png")
            print("  → visual_roc_curve.png")
            print("  → phish360_efficientnet_b0_best.pt")
            print("Goodbye!")
            break

        if not os.path.exists(user_input):
            print(f"File not found: {user_input}")
            continue

        try:
            predict_image(user_input)
        except Exception as e:
            print(f"Error processing image: {e}")

# ─────────────────────────────────────────
# STILL TO DO (not covered by this draft)
# ─────────────────────────────────────────
# 1. This module currently classifies a screenshot in isolation. The
#    report's visual module also planned perceptual hashing (imagehash)
#    against a trusted-brand reference set as a fast pre-filter, and
#    Tesseract OCR for brand-text extraction — neither is included
#    here yet, since this draft focuses on the CNN classifier itself.
# 2. Live inference (a real URL → screenshot → this model) still needs
#    the Playwright screenshot step from the URL module connected as
#    input here — see the "STILL TO DO" note in url_lightgbm_classifier.py.