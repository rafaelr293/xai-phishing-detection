"""
Visual layer — EXPLAINABILITY
Grad-CAM: làm nổi bật vùng của screenshot mà model tập trung vào khi dự đoán
(vd form đăng nhập giả, logo). Tương đương SHAP của các module email / URL.
"""

import os

import matplotlib
import numpy as np
import torch
from PIL import Image

from Visual_Model import IMG_SIZE, eval_transform


class GradCAM:
    """Grad-CAM trên một layer conv. Mặc định dùng khối conv cuối của EfficientNet-B0.

    Dùng tensor hook (register_hook trên output) thay vì module backward hook để an toàn với
    các phép in-place (SiLU inplace) trong EfficientNet.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.features[-1]
        self.activations = None
        self.gradients = None
        self._handle = self.target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, 'gradients', g.detach()))

    def remove(self):
        self._handle.remove()

    def generate(self, input_tensor, class_idx=1):
        """input_tensor [1,3,H,W] -> (heatmap [h,w] trong [0,1], probs [2])."""
        self.model.eval()
        self.gradients = None
        with torch.enable_grad():
            output = self.model(input_tensor)
            self.model.zero_grad()
            output[0, class_idx].backward()
        probs = torch.softmax(output.detach(), dim=1)[0].cpu().numpy()

        weights = self.gradients.mean(dim=(0, 2, 3))                 # [C]
        cam = (self.activations[0] * weights[:, None, None]).mean(dim=0)
        heatmap = np.maximum(cam.cpu().numpy(), 0)
        heatmap /= (heatmap.max() + 1e-8)
        return heatmap, probs


def explain_image(gradcam, image_path, device, save_path, class_idx=1, show=False, title=None):
    """Chạy Grad-CAM cho một ảnh, lưu overlay (gốc | heatmap) vào save_path.

    Trả về (probs, save_path). class_idx=1 -> vùng ảnh ủng hộ nhãn "phishing".
    """
    import matplotlib.pyplot as plt  # import muộn để backend (Agg/GUI) do caller chọn

    img = Image.open(image_path).convert('RGB')
    tensor = eval_transform(img).unsqueeze(0).to(device)
    heatmap, probs = gradcam.generate(tensor, class_idx=class_idx)
    heatmap_resized = np.array(Image.fromarray(heatmap).resize((IMG_SIZE, IMG_SIZE)))
    base = img.resize((IMG_SIZE, IMG_SIZE))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(base)
    axes[0].set_title('Original Screenshot')
    axes[0].axis('off')
    axes[1].imshow(base)
    axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.5)
    axes[1].set_title(title or 'Grad-CAM — Model Focus Area')
    axes[1].axis('off')
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return probs, save_path


def use_headless_backend():
    """Dùng backend không cần GUI (chỉ lưu file)."""
    matplotlib.use('Agg')
