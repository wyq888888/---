"""
茎秆边缘检测预处理预览脚本
用法: python stem_edge_preview.py <图片路径>
      python stem_edge_preview.py  (不传参数则打开文件选择框)

输出: 6格对比窗口，按任意键切换，按 Q 退出
      同时保存 edge_result.png 到脚本同目录
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

from tape_pre import detect_tape_edges

DEFAULT_WEIGHTS = Path(__file__).with_name("tape.pt")

# ─────────────────────────────────────────────
# 可调参数（先用这组，效果不好再改）
# ─────────────────────────────────────────────
BILATERAL_D         = 9       # 双边滤波邻域直径
BILATERAL_SIGMA_C   = 75      # 颜色空间sigma
BILATERAL_SIGMA_S   = 75      # 空间sigma

CANNY_LOW           = 30      # Canny低阈值
CANNY_HIGH          = 90      # Canny高阈值

MORPH_KERNEL_H      = 11      # 竖向形态学核高度（连通断点用）
MEDIAN_KERNEL_SIZE  = 3       # ROI中值滤波核大小（必须为奇数）
# ─────────────────────────────────────────────


def load_image(path=None):
    if path:
        img = cv2.imread(path)
        if img is None:
            print(f"[错误] 无法读取图片: {path}")
            sys.exit(1)
        return img
    # 没传路径则弹文件选择（需要tkinter）
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        root.destroy()
        if not path:
            print("[错误] 未选择图片")
            sys.exit(1)
        img = cv2.imread(path)
        if img is None:
            print(f"[错误] 无法读取图片: {path}")
            sys.exit(1)
        return img
    except Exception as e:
        print(f"[错误] 请直接传入图片路径: python stem_edge_preview.py <路径>\n{e}")
        sys.exit(1)


def _shift_edge_down(points, shift_pixels, image_height):
    shifted = []
    for x, y in points:
        shifted_y = max(0.0, min(float(image_height - 1), float(y) + float(shift_pixels)))
        shifted.append((int(round(float(x))), int(round(shifted_y))))
    return shifted


def draw_shifted_bottom_edge(image, tape_edges, shift_pixels=10):
    overlay = image.copy()
    image_height = overlay.shape[0]

    if not tape_edges:
        return overlay

    item = tape_edges[0]
    bottom_points = item["bottom_edge"]["points"]
    shifted_points = _shift_edge_down(bottom_points, shift_pixels, image_height)
    if len(shifted_points) == 2:
        cv2.line(overlay, shifted_points[0], shifted_points[1], (0, 0, 255), 2, cv2.LINE_AA)

    return overlay


def _clamp_int(value, low, high):
    return max(low, min(high, int(value)))


def crop_roi_from_edges(image, top_points, shifted_points, padding_x=20, padding_y=20):
    xs = [point[0] for point in top_points] + [point[0] for point in shifted_points]
    ys = [point[1] for point in top_points] + [point[1] for point in shifted_points]

    x1 = _clamp_int(min(xs) - padding_x, 0, image.shape[1] - 1)
    x2 = _clamp_int(max(xs) + padding_x, 0, image.shape[1] - 1)
    y1 = _clamp_int(min(ys) - padding_y, 0, image.shape[0] - 1)
    y2 = _clamp_int(max(ys) + padding_y, 0, image.shape[0] - 1)

    if x2 <= x1:
        x2 = min(image.shape[1] - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(image.shape[0] - 1, y1 + 1)

    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def preprocess_roi(roi_image):
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, MEDIAN_KERNEL_SIZE)
    scharr_x = cv2.Scharr(denoised, cv2.CV_64F, dx=1, dy=0)
    scharr_x = cv2.convertScaleAbs(scharr_x)
    return gray, denoised, scharr_x


def preprocess(image):
    results = {}
    results["①原图"] = image.copy()

    # ── Step 1: 绿色通道提取（番茄枝干对比度最好）──
    gray_g = image[:, :, 1]   # BGR → G通道
    gray_bgr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 同时保留普通灰度作对比，取绿色通道
    gray = gray_g
    results["②绿色通道灰度"] = gray

    # ── Step 2: 双边滤波（保边降噪）──
    filtered = cv2.bilateralFilter(
        gray,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_C,
        sigmaSpace=BILATERAL_SIGMA_S
    )
    results["③双边滤波"] = filtered

    # ── Step 3: Scharr 竖直方向边缘增强 ──
    scharr_x = cv2.Scharr(filtered, cv2.CV_64F, dx=1, dy=0)
    scharr_x = cv2.convertScaleAbs(scharr_x)
    results["④Scharr竖直边缘"] = scharr_x

    # ── Step 4: Canny 边缘检测 ──
    edges = cv2.Canny(scharr_x, CANNY_LOW, CANNY_HIGH)
    results["⑤Canny边缘"] = edges

    # ── Step 5: 竖向形态学闭运算（连通断点）──
    kernel_v = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, MORPH_KERNEL_H)
    )
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)
    results["⑥闭运算后(最终)"] = edges_closed

    return results


def make_grid(results, target_h=600):
    """把所有步骤拼成2行3列的预览图"""
    panels = []
    for title, img in results.items():
        # 统一缩放到 target_h 高度
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_h))

        # 灰度图转BGR方便拼接
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

        # 加标题栏
        bar = np.zeros((36, new_w, 3), dtype=np.uint8)
        cv2.putText(bar, title, (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
        panel = np.vstack([bar, resized])
        panels.append(panel)

    # 补齐到6格（2行3列）
    while len(panels) % 3 != 0:
        panels.append(np.zeros_like(panels[0]))

    rows = []
    for i in range(0, len(panels), 3):
        row = np.hstack(panels[i:i+3])
        rows.append(row)
    grid = np.vstack(rows)
    return grid


def main():
    parser = argparse.ArgumentParser(description="茎秆边缘检测预处理预览脚本")
    parser.add_argument("-rgb", help="RGB图片路径")
    parser.add_argument("-depth", help="对应深度文件路径（npy 或 16位深度图）")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="tape.pt 权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    args = parser.parse_args()

    tape_result = detect_tape_edges(args.rgb, args.depth, weights_path=args.weights, conf=args.conf, imgsz=args.imgsz)
    tape_edges = tape_result["detections"]
    image = tape_result["filtered_rgb"]

    if tape_edges:
        print("[tape] 识别到目标，输出下边/左边两点：")
        print(f"  - 目标#{tape_edges[0]['index']} bottom={tape_edges[0]['bottom_edge']['points']} left={tape_edges[0]['left_edge']['points']}")
        if len(tape_edges) > 1:
            print(f"[tape] 检测到 {len(tape_edges)} 个目标，当前仅绘制第一个目标的底边")
    else:
        print("[tape] 未识别到目标")

    preview_image = draw_shifted_bottom_edge(image, tape_edges, shift_pixels=10) if tape_edges else image.copy()

    roi_edge_window = None
    roi_gray = None
    roi_denoised = None
    roi_edges = None
    roi_bbox = None
    if tape_edges:
        bottom_points = tape_edges[0]["bottom_edge"]["points"]
        shifted_points = _shift_edge_down(bottom_points, 10, image.shape[0])
        roi_image, roi_bbox = crop_roi_from_edges(image, bottom_points, shifted_points, padding_x=15, padding_y=10)
        if roi_image.size != 0:
            roi_gray, roi_denoised, roi_edges = preprocess_roi(roi_image)
            roi_edge_window = cv2.resize(
                roi_edges,
                None,
                fx=4.0,
                fy=4.0,
                interpolation=cv2.INTER_CUBIC,
            )

    print(f"[信息] 图片尺寸: {image.shape[1]}×{image.shape[0]}")
    print(f"[参数] 仅绘制下边向下平移10像素后的红线")

    # 保存结果
    save_path = "edge_result.png"
    cv2.imwrite(save_path, preview_image)
    print(f"[保存] 结果已写入: {save_path}")

    # 显示窗口
    win = "茎秆边缘检测 — 平移边线预览 | 按 Q 退出 | 按 S 保存当前帧"
    cv2.imshow(win, preview_image)

    if roi_edge_window is not None:
        roi_win = "ROI竖直边缘提取(放大)"
        cv2.imshow(roi_win, roi_edge_window)
        if roi_bbox is not None:
            print(f"[ROI] 裁剪区域: x1={roi_bbox[0]} y1={roi_bbox[1]} x2={roi_bbox[2]} y2={roi_bbox[3]}")
        if roi_gray is not None:
            print(f"[ROI] 尺寸: {roi_gray.shape[1]}×{roi_gray.shape[0]}，已放大后展示")

    print("[提示] 窗口已打开，按 Q 退出，按 S 另存图片")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            out = f"edge_result_saved.png"
            cv2.imwrite(out, preview_image)
            print(f"[保存] 另存为: {out}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
