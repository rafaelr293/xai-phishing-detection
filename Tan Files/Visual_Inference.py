"""
Visual layer — INFERENCE
Nạp model đã train và phân loại screenshot (legitimate / phishing) từ terminal.

Lưu ý: visual layer nhận screenshot CÓ SẴN, chưa tự mở URL để chụp màn hình.
Kết nối Playwright (URL -> screenshot) là việc tùy chọn khi tích hợp meta layer.

Cách dùng:
  python visual_layer_inference.py --image "D:\\screenshots\\page1.png"
  python visual_layer_inference.py --image "D:\\screenshots\\page1.png" "D:\\screenshots\\page2.png"
  python visual_layer_inference.py --interactive
  python visual_layer_inference.py --image "D:\\screenshots\\page1.png" --gradcam
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch

from Visual_Model import CLASS_NAMES, eval_transform, load_checkpoint
from PIL import Image

DEFAULT_MODEL = os.path.join('models', 'phish360_efficientnet_b0_best.pt')


def risk_level(p):
    # Ngưỡng tạm — cần hiệu chỉnh lại sau, giống hai module còn lại.
    if p >= 0.85:
        return 'CRITICAL'
    if p >= 0.60:
        return 'HIGH'
    if p >= 0.35:
        return 'MEDIUM'
    return 'LOW'


def resolve_model_path(path):
    """Cho phép checkpoint cũ nằm ngay thư mục hiện tại (tên mặc định cũ)."""
    if os.path.exists(path):
        return path
    legacy = os.path.basename(DEFAULT_MODEL)
    if path == DEFAULT_MODEL and os.path.exists(legacy):
        return legacy
    return path


def predict(model, image_path, device, threshold=0.5):
    img = Image.open(image_path).convert('RGB')
    tensor = eval_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    p_phish = float(probs[1])
    return {'image': image_path,
            'label': 'phishing' if p_phish >= threshold else 'legitimate',
            'phishing_probability': round(p_phish, 4),
            'legitimate_probability': round(float(probs[0]), 4),
            'threshold': threshold,
            'risk_level': risk_level(p_phish),
            'gradcam_path': ''}


def scan_image(model, path, device, args, gradcam=None, counter=0):
    path = path.strip().strip('"').strip("'")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    try:
        r = predict(model, path, device, args.threshold)
        if gradcam is not None:
            from Visual_Explainability import explain_image
            stem = os.path.splitext(os.path.basename(path))[0]
            out = os.path.join(args.gradcam_dir, f"{counter:03d}_{stem}_gradcam.png")
            explain_image(gradcam, path, device, out, class_idx=1, show=args.show)
            r['gradcam_path'] = out
    except Exception as e:
        print(f"Error processing image '{path}': {type(e).__name__}: {e}")
        return None
    print_result(r)
    return r


def print_result(r):
    mark = '🚨 PHISHING' if r['label'] == 'phishing' else '✅ LEGITIMATE'
    print(f"\n{'─' * 55}")
    print(f"Image  : {r['image']}")
    print(f"Result : {mark}")
    print(f"Legitimate probability : {r['legitimate_probability'] * 100:.2f}%")
    print(f"Phishing   probability : {r['phishing_probability'] * 100:.2f}%")
    print(f"Threshold              : {r['threshold']}")
    print(f"Risk Level             : {r['risk_level']}")
    if r['gradcam_path']:
        print(f"Grad-CAM               : {r['gradcam_path']}")
    print('─' * 55)


def main():
    ap = argparse.ArgumentParser(description="Visual layer — phân loại screenshot bằng model đã train")
    ap.add_argument('--image', nargs='+', metavar='PATH', help="một hoặc nhiều screenshot")
    ap.add_argument('--file', help="file text, mỗi dòng một đường dẫn ảnh")
    ap.add_argument('--interactive', action='store_true',
                    help="nhập đường dẫn liên tục (gõ 'quit' để thoát)")
    ap.add_argument('--gradcam', action='store_true', help="tạo ảnh Grad-CAM giải thích cho mỗi screenshot")
    ap.add_argument('--gradcam-dir', default='gradcam_outputs', help="thư mục lưu Grad-CAM")
    ap.add_argument('--show', action='store_true', help="hiện Grad-CAM trên màn hình (ngoài việc lưu file)")
    ap.add_argument('--csv-out', help="lưu kết quả ra CSV")
    ap.add_argument('--json-out', help="lưu kết quả ra JSON")
    ap.add_argument('--model', default=DEFAULT_MODEL, help=f"mặc định: {DEFAULT_MODEL}")
    ap.add_argument('--threshold', type=float, default=0.5,
                    help="phishing nếu P(phishing) >= threshold (mặc định 0.5 = argmax)")
    args = ap.parse_args()

    if not (args.image or args.file or args.interactive):
        ap.error("cần ít nhất một trong: --image, --file, --interactive")

    if args.gradcam and not args.show:
        from Visual_Explainability import use_headless_backend
        use_headless_backend()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = load_checkpoint(resolve_model_path(args.model), device)
    except FileNotFoundError as e:
        sys.exit(str(e))

    gradcam = None
    if args.gradcam:
        from Visual_Explainability import GradCAM
        gradcam = GradCAM(model)

    results, counter = [], 0
    paths = list(args.image or [])
    if args.file:
        if not os.path.exists(args.file):
            sys.exit(f"Không tìm thấy file '{args.file}'.")
        with open(args.file, encoding='utf-8-sig') as f:
            paths += [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith('#')]

    for p in paths:
        r = scan_image(model, p, device, args, gradcam, counter)
        counter += 1
        if r:
            results.append(r)

    if args.interactive:
        print("\n" + "=" * 55 + "\n     INTERACTIVE WEBPAGE SCREENSHOT TESTER\n" + "=" * 55)
        print("Nhập đường dẫn tới screenshot (.png/.jpg). 'quit' để thoát.")
        while True:
            try:
                line = input("\nImage path (or 'quit'): ").strip().strip('"').strip("'")
            except EOFError:
                break
            if line.lower() in ('quit', 'exit', 'q'):
                break
            if not line:
                continue
            r = scan_image(model, line, device, args, gradcam, counter)
            counter += 1
            if r:
                results.append(r)

    if results:
        n_ph = sum(r['label'] == 'phishing' for r in results)
        print(f"\nTổng: {len(results)} ảnh | phishing: {n_ph} | legitimate: {len(results) - n_ph}")
        if args.csv_out:
            pd.DataFrame(results).to_csv(args.csv_out, index=False, encoding='utf-8-sig')
            print(f"Saved -> {args.csv_out}")
        if args.json_out:
            with open(args.json_out, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved -> {args.json_out}")
    if gradcam is not None:
        gradcam.remove()


if __name__ == '__main__':
    main()
