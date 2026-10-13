"""
Visual layer — TRAIN
Kiểm tra ảnh hỏng, chia train/val/test, training, đánh giá và lưu model.

Usage:
  python visual_layer_train.py --data-dir phish360-data
  python visual_layer_train.py --data-dir phish360-data --epochs 10 --num-workers 2

Lưu:
  models/phish360_efficientnet_b0_best.pt   (state_dict tốt nhất theo val loss)
  models/visual_metadata.json               (cấu hình + metric test)
  visual_confusion_matrix.png, visual_roc_curve.png
"""

import argparse
import concurrent.futures
import json
import os
import sys
import traceback
from datetime import datetime, timezone

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.multiprocessing
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_curve, auc)
from torch.utils.data import DataLoader, Subset, random_split

from Visual_Model import (IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES,
                                Phish360Dataset, build_model, eval_transform,
                                load_phish360, train_transform)

# Windows: lỗi "Couldn't open shared file mapping ... 1455" (hết virtual memory / page file)
# do DataLoader worker. Giữ số worker thấp + không giữ worker sống cho test loader là cách
# sửa thực sự; dòng dưới là no-op trên Windows nhưng vô hại, có ích trên Linux.
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass


# ─────────────────────────────────────────
# IMAGE CHECK
# (module-level để pickle được khi ProcessPoolExecutor spawn trên Windows)
# ─────────────────────────────────────────
def _check_one_image(args):
    """Nhận (idx, path), trả (idx, path, error_string_hoặc_None)."""
    idx, path = args
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"file not found: {path}")
        with Image.open(path) as img:       # verify(): kiểm tra cấu trúc nhanh
            img.verify()
        with Image.open(path) as img:       # mở lại + decode thật để bắt các lỗi verify() bỏ sót
            img.convert('RGB')
        return (idx, path, None)
    except Exception as e:
        return (idx, path, f"{type(e).__name__}: {e}")


def scan_corrupted_images(df, path_col='full_image_path', verbose=True, num_workers=8):
    """Mở mọi ảnh trong df[path_col] để bắt file thiếu/0 byte/cắt cụt/HTML-lưu-thành-.png...
    Trả về danh sách index dataframe bị lỗi."""
    total = len(df)
    tasks = list(zip(df.index, df[path_col]))
    bad_indices, done = [], 0
    if verbose:
        print(f"\nScanning {total:,} images for corruption using {num_workers} processes...")
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


# ─────────────────────────────────────────
# TRAIN / EVAL HELPERS
# ─────────────────────────────────────────
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
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def plot_confusion_matrix(cm_data, title, filename, show=False):
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 14})
    plt.title(title, fontsize=13)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"Saved → {filename}")


def plot_roc(y_true, y_scores, title, filename, show=False):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"Saved → {filename}")
    return roc_auc


# ─────────────────────────────────────────
# MAIN  (khối __main__ BẮT BUỘC trên Windows khi num_workers > 0)
# ─────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Visual layer — train EfficientNet-B0 on Phish360")
    ap.add_argument('--data-dir', default='phish360-data')
    ap.add_argument('--model-dir', default='models')
    ap.add_argument('--epochs', type=int, default=3,
                    help="mặc định 3 (chạy thử pipeline); tăng lên ~10+ khi cần độ chính xác")
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=2,
                    help="DataLoader workers; nếu vẫn lỗi 1455 trên Windows hãy hạ về 0")
    ap.add_argument('--scan-workers', type=int, default=8)
    ap.add_argument('--skip-scan', action='store_true', help="bỏ qua bước kiểm tra ảnh hỏng")
    ap.add_argument('--show', action='store_true', help="hiện biểu đồ (plt.show) ngoài việc lưu")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use('Agg')

    print("=" * 55)
    print("   PHISHING WEBPAGE CLASSIFIER — EfficientNet-B0 (TRAIN)")
    print("=" * 55)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    torch.set_num_threads(8)

    print("\nLoading dataset...")
    df = load_phish360(data_dir=args.data_dir)

    # Quét ảnh hỏng mỗi lần chạy để danh sách loại trừ luôn khớp với dữ liệu hiện tại
    if not args.skip_scan:
        bad = scan_corrupted_images(df, num_workers=args.scan_workers)
        if bad:
            print(f"Excluding {len(bad)} corrupted sample(s) from training")
            df = df.drop(index=bad).reset_index(drop=True)
        else:
            print("No corrupted images found.")

    phishing_count = int(df['label_num'].sum())
    legit_count = len(df) - phishing_count
    print(f"\n  Total samples      : {len(df):,}")
    print(f"  Phishing samples   : {phishing_count:,}  ({phishing_count/len(df)*100:.1f}%)")
    print(f"  Legitimate samples : {legit_count:,}  ({legit_count/len(df)*100:.1f}%)")

    # Bỏ các cột nặng (full_html, URL...) — nếu giữ, mỗi worker Windows phải nhận bản sao
    # toàn bộ HTML thô, dễ vượt 2GB và gây OSError.
    df = df[['full_image_path', 'label_num']].copy()

    # ─── split 70 / 15 / 15 (seed 42) ───
    n = len(df)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    tr, va, te = random_split(range(n), [train_size, val_size, test_size],
                              generator=torch.Generator().manual_seed(42))
    # train có augmentation, val/test thì không
    train_subset = Subset(Phish360Dataset(df, transform=train_transform), tr.indices)
    val_subset = Subset(Phish360Dataset(df, transform=eval_transform), va.indices)
    test_subset = Subset(Phish360Dataset(df, transform=eval_transform), te.indices)

    nw = args.num_workers
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, persistent_workers=(nw > 0))
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, persistent_workers=(nw > 0))
    # test_loader chỉ dùng sau training -> num_workers=0 để không giữ worker/shared memory
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nTraining samples  : {train_size:,}")
    print(f"Validation samples: {val_size:,}")
    print(f"Testing samples   : {test_size:,}")

    # ─── model ───
    model = build_model(device, pretrained=True)
    # Full fine-tuning. Nếu quá chậm / overfit, có thể freeze model.features, chỉ train classifier.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    try:
        import psutil
        has_psutil = True
    except ImportError:
        has_psutil = False
        print("(tip: `pip install psutil` để xem RAM trống mỗi epoch, hữu ích khi chẩn đoán crash im lặng)")

    os.makedirs(args.model_dir, exist_ok=True)
    ckpt_path = os.path.join(args.model_dir, 'phish360_efficientnet_b0_best.pt')
    meta_path = os.path.join(args.model_dir, 'visual_metadata.json')

    print("\nTraining model...")
    best_val_loss, best_epoch = float('inf'), 0
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer, True)
            val_loss, val_acc = run_epoch(model, val_loader, device, criterion, train_mode=False)
            scheduler.step(val_loss)

            line = (f"Epoch {epoch:2d}/{args.epochs} | "
                    f"Train loss: {train_loss:.4f} acc: {train_acc*100:.2f}% | "
                    f"Val loss: {val_loss:.4f} acc: {val_acc*100:.2f}%")
            if has_psutil:
                mem = psutil.virtual_memory()
                line += f" | RAM free: {mem.available / 1e9:.2f} GB ({mem.percent:.0f}% used)"
            print(line)

            if val_loss < best_val_loss:
                best_val_loss, best_epoch = val_loss, epoch
                torch.save(model.state_dict(), ckpt_path)
                print(f"  → New best model saved (val loss {val_loss:.4f})")
    except Exception:
        print("\n" + "=" * 55 + "\nTRAINING CRASHED — full traceback below:\n" + "=" * 55)
        traceback.print_exc()
        sys.stdout.flush()
        raise

    print("\nDone! Loading best checkpoint for evaluation...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    # ─── evaluate on test set ───
    model.eval()
    preds_l, labels_l, probs_l = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            probs_l.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            preds_l.extend(outputs.argmax(dim=1).cpu().numpy())
            labels_l.extend(labels.numpy())
    all_preds, all_labels, all_probs = map(np.array, (preds_l, labels_l, probs_l))

    acc = accuracy_score(all_labels, all_preds)
    print("\n" + "=" * 55 + "\nRESULTS — Phish360 (test set)\n" + "=" * 55)
    print(f"Accuracy : {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue  Negatives (Legit → Legit)     : {tn}  ✅")
    print(f"True  Positives (Phish → Phish)     : {tp}  ✅")
    print(f"False Positives (Legit → Phish)     : {fp}  ⚠️")
    print(f"False Negatives (Phish → Legit)     : {fn}  🚨  (missed phishing — check this closely)")

    plot_confusion_matrix(cm, 'Confusion Matrix — Phish360', 'visual_confusion_matrix.png', args.show)
    roc_auc = plot_roc(all_labels, all_probs, 'ROC Curve — Phish360', 'visual_roc_curve.png', args.show)

    meta = {'version': 'visual_layer_efficientnet_b0',
            'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'checkpoint': os.path.basename(ckpt_path), 'img_size': IMG_SIZE,
            'mean': IMAGENET_MEAN, 'std': IMAGENET_STD, 'class_names': CLASS_NAMES,
            'epochs': args.epochs, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
            'split': {'train': train_size, 'val': val_size, 'test': test_size, 'seed': 42},
            'test': {'accuracy': float(acc), 'roc_auc': float(roc_auc),
                     'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}}
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nFiles saved:\n  → {ckpt_path}\n  → {meta_path}\n"
          f"  → visual_confusion_matrix.png\n  → visual_roc_curve.png")


if __name__ == "__main__":
    main()
