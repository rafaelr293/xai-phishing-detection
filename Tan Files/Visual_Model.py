"""
Visual layer — MODEL
Kiến trúc EfficientNet-B0, transforms và dataset loader (dùng chung cho train / inference /
explainability).

Phải nằm ở module level (không đặt trong khối __main__) để các DataLoader worker trên Windows
import được Phish360Dataset khi spawn.
"""

import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ['Legitimate', 'Phishing']          # index 0 = legit, 1 = phishing

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.2),   # augmentation nhẹ — trang login hiếm khi bị lật gương
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────
def load_phish360(data_dir=None):
    """
    Nạp Phish360 — cấu trúc:
      Phish360_legit.parquet / Phish360_phish.parquet, mỗi file có các cột:
      sample_id, URL, full_html, Class, image_path, raw_label
      'image_path' là đường dẫn TƯƠNG ĐỐI so với data_dir (vd 'screenshots/L00003_legitimate.png').

    data_dir mặc định là thư mục chứa file này.
    """
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))

    frames = []
    for fname, label_val in [('Phish360_legit.parquet', 0), ('Phish360_phish.parquet', 1)]:
        full_path = os.path.join(data_dir, fname)
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Could not find '{fname}' in {data_dir}. "
                f"Either move the data there, or pass data_dir explicitly to load_phish360()."
            )
        part = pd.read_parquet(full_path)
        part['label_num'] = label_val        # legit=0 / phish=1, biết từ file nguồn
        frames.append(part)
        print(f"  Loaded {fname} → {len(part):,} samples")

    df = pd.concat(frames, ignore_index=True)
    if 'image_path' not in df.columns:
        raise KeyError(f"Expected an 'image_path' column. Found columns: {list(df.columns)}")

    df['full_image_path'] = df['image_path'].apply(lambda p: os.path.join(data_dir, p))
    # NOTE: cột 'full_html' vẫn có sẵn nếu sau này muốn kết hợp HTML + ảnh.
    print(f"\n  Total samples → {len(df):,}")
    return df


class Phish360Dataset(Dataset):
    """Mở từng screenshot từ cột 'full_image_path' (đã là đường dẫn tuyệt đối)."""

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


# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
def build_model(device, pretrained=True):
    """EfficientNet-B0, thay lớp classifier cuối thành 2 lớp (legit / phishing).

    pretrained=True  : khởi tạo bằng ImageNet weights (dùng khi train)
    pretrained=False : không tải weights (dùng khi inference — checkpoint sẽ ghi đè)
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))
    return model.to(device)


def load_checkpoint(path, device):
    """Nạp state_dict đã train vào EfficientNet-B0 và chuyển sang eval()."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint '{path}'. Hãy chạy visual_layer_train.py trước "
            f"hoặc chỉ đường dẫn bằng --model."
        )
    model = build_model(device, pretrained=False)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_screenshot(path):
    """Đọc một screenshot -> (PIL RGB image, tensor [1,3,H,W] đã normalise)."""
    img = Image.open(path).convert('RGB')
    return img, eval_transform(img).unsqueeze(0)
