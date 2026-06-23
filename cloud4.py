#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""RealSense D435 bag/live YOLO depth sampler.

使用说明：
- 支持命令--bag从 .bag 文件读取，如果不指定--bag就直接打开 RealSense 实时相机读取
- 使用指定的 YOLO 模型对 RGB 流做目标识别
- 稳定性过滤：连续3帧同一目标IOU>0.7才触发精确测量，取中间帧计算最小外接矩形直径
- 计算每个目标轮廓的最小外接矩形短边像素长度，并换算物理直径
- 保存 [x, y, z, min_diameter_px, physical_diameter_m, physical_diameter_cm] 矩阵，以及带元数据的 CSV
"""

import argparse
import csv
import ctypes
import os
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# TODO: 替换你的路径
DEFAULT_WEIGHTS = r"C:\Users\买房 人生大计\Desktop\小番茄\size.pt"
DEFAULT_OUTPUT_DIR = r"C:\Users\买房 人生大计\Desktop\小番茄\xyz_output"


def _to_windows_short_path(path: Path) -> Path:
	if os.name != "nt":
		return path
	if not path.exists():
		return path
	if path.as_posix().isascii():
		return path
	try:
		buffer = ctypes.create_unicode_buffer(32768)
		result = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
		if result == 0:
			return path
		short_path = Path(buffer.value)
		return short_path if short_path.exists() else path
	except Exception:
		return path


def _prepare_output_dir(output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)


def _setup_pipeline(bag_path: Optional[Path], width: int, height: int, fps: int) -> Tuple[rs.pipeline, rs.pipeline_profile]:
	pipeline = rs.pipeline()
	cfg = rs.config()

	if bag_path is not None:
		bag_path = _to_windows_short_path(bag_path)
		cfg.enable_device_from_file(str(bag_path), repeat_playback=False)
		print(f"[INFO] 读取 bag: {bag_path}")
	else:
		cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
		cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
		print(f"[INFO] 打开实时相机: {width}x{height} @ {fps} FPS")

	profile = pipeline.start(cfg)
	device = profile.get_device()
	if bag_path is not None:
		playback = device.as_playback()
		playback.set_real_time(False)
		if hasattr(playback, "set_repeat_playback"):
			playback.set_repeat_playback(False)

	return pipeline, profile


def _frame_to_bgr(color_frame: rs.video_frame) -> np.ndarray:
	frame_data = np.asanyarray(color_frame.get_data())
	frame_format = color_frame.profile.format()
	if frame_format == rs.format.rgb8:
		return cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
	return frame_data


def _sample_depth_meters(depth_frame: rs.depth_frame, x: int, y: int, radius: int = 2) -> float:
	width = depth_frame.get_width()
	height = depth_frame.get_height()
	for r in range(radius + 1):
		x_min = max(0, x - r)
		x_max = min(width - 1, x + r)
		y_min = max(0, y - r)
		y_max = min(height - 1, y + r)
		for yy in range(y_min, y_max + 1):
			for xx in range(x_min, x_max + 1):
				z = float(depth_frame.get_distance(int(xx), int(yy)))
				if z > 0:
					return z
	return float("nan")


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
	"""计算两个 [x1,y1,x2,y2] 框的 IOU。"""
	ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
	ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
	inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
	if inter == 0:
		return 0.0
	area_a = (a[2] - a[0]) * (a[3] - a[1])
	area_b = (b[2] - b[0]) * (b[3] - b[1])
	return inter / (area_a + area_b - inter)


def _match_boxes(boxes_a: np.ndarray, boxes_b: np.ndarray, iou_thresh: float = 0.7) -> List[Tuple[int, int]]:
	"""
	贪心匹配两帧检测框，返回 (idx_a, idx_b) 配对列表（IOU >= iou_thresh 才配对）。
	boxes 格式：(N, 4) xyxy。
	"""
	if len(boxes_a) == 0 or len(boxes_b) == 0:
		return []
	matched = []
	used_b = set()
	for i, a in enumerate(boxes_a):
		best_iou, best_j = 0.0, -1
		for j, b in enumerate(boxes_b):
			if j in used_b:
				continue
			iou = _box_iou(a, b)
			if iou > best_iou:
				best_iou, best_j = iou, j
		if best_j >= 0 and best_iou >= iou_thresh:
			matched.append((i, best_j))
			used_b.add(best_j)
	return matched


def _draw_box_label(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str) -> None:
	color = (0, 255, 0)
	text_color = (255, 255, 255)
	bg_color = (0, 128, 0)
	cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.6
	thickness = 1
	(text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
	text_x = max(0, x1)
	text_y = max(text_height + baseline + 4, y1 - 6)
	bg_top = max(0, text_y - text_height - baseline - 4)
	bg_bottom = min(image.shape[0] - 1, text_y + baseline)
	bg_right = min(image.shape[1] - 1, text_x + text_width + 8)
	cv2.rectangle(image, (text_x, bg_top), (bg_right, bg_bottom), bg_color, -1)
	cv2.putText(image, label, (text_x + 4, text_y - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)


def _min_area_rect_short_edge_px(roi_bgr: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
	"""
	从ROI中分割前景目标，返回最小外接矩形短边像素长度。
	分割策略（按优先级依次尝试）：
	  1. GrabCut：以ROI中心20%区域为确定前景种子，边缘10%为确定背景
	  2. HSV颜色分割（红/橙/绿番茄备用）
	  3. Otsu兜底
	最终对轮廓取凸包再做minAreaRect，消除花萼/茎秆凹陷对旋转角度的干扰。
	"""
	h, w = roi_bgr.shape[:2]
	if h < 10 or w < 10:
		return float("nan"), None, None

	kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

	def _best_contour_from_mask(mask: np.ndarray):
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if not contours:
			return None, 0.0, mask
		c = max(contours, key=cv2.contourArea)
		return c, float(cv2.contourArea(c)), mask

	roi_area = float(h * w)
	best_contour = None
	best_area    = 0.0
	best_mask    = None

	# ── 策略1：GrabCut ──────────────────────────────────────────────
	try:
		gc_mask = np.zeros((h, w), np.uint8)
		cy0 = int(h * 0.40); cy1 = int(h * 0.60)
		cx0 = int(w * 0.40); cx1 = int(w * 0.60)
		gc_mask[cy0:cy1, cx0:cx1] = cv2.GC_FGD
		border = max(2, int(min(h, w) * 0.10))
		gc_mask[:border,  :]  = cv2.GC_BGD
		gc_mask[-border:, :]  = cv2.GC_BGD
		gc_mask[:,  :border]  = cv2.GC_BGD
		gc_mask[:, -border:]  = cv2.GC_BGD
		gc_mask[(gc_mask != cv2.GC_FGD) & (gc_mask != cv2.GC_BGD)] = cv2.GC_PR_FGD
		bgd_model = np.zeros((1, 65), np.float64)
		fgd_model = np.zeros((1, 65), np.float64)
		cv2.grabCut(roi_bgr, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
		fg_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
		c, area, m = _best_contour_from_mask(fg_mask)
		if c is not None and roi_area * 0.15 < area < roi_area * 0.90:
			best_contour, best_area, best_mask = c, area, m
	except Exception:
		pass

	# ── 策略2：HSV颜色分割 ───────────────────────────────────────────
	if best_contour is None or best_area < roi_area * 0.10:
		hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
		m1 = cv2.inRange(hsv, (0,   60, 60), (15,  255, 255))
		m2 = cv2.inRange(hsv, (160, 60, 60), (180, 255, 255))
		m3 = cv2.inRange(hsv, (10,  80, 80), (30,  255, 255))
		m4 = cv2.inRange(hsv, (35,  60, 60), (85,  255, 255))
		color_mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), cv2.bitwise_or(m3, m4))
		c, area, m = _best_contour_from_mask(color_mask)
		if c is not None and area > best_area and roi_area * 0.10 < area < roi_area * 0.95:
			best_contour, best_area, best_mask = c, area, m

	# ── 策略3：Otsu兜底 ─────────────────────────────────────────────
	if best_contour is None or best_area < roi_area * 0.08:
		gray    = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		for flag in (cv2.THRESH_BINARY + cv2.THRESH_OTSU, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU):
			_, otsu_mask = cv2.threshold(blurred, 0, 255, flag)
			c, area, m = _best_contour_from_mask(otsu_mask)
			if c is not None and area > best_area:
				best_contour, best_area, best_mask = c, area, m

	if best_contour is None or best_area <= 0:
		return float("nan"), None, None

	# ── 凸包 → minAreaRect ─────────────────────────────────────────
	hull = cv2.convexHull(best_contour)
	if hull is None or len(hull) < 3:
		hull = best_contour
	rect = cv2.minAreaRect(hull)
	short_edge_px = float(min(rect[1]))
	if short_edge_px <= 0:
		return float("nan"), hull, best_mask

	return short_edge_px, hull, best_mask


def _compute_diameters_on_frame(
	xyxy: np.ndarray,
	xywh: np.ndarray,
	cls_ids: Optional[np.ndarray],
	confs: Optional[np.ndarray],
	depth_frame: rs.depth_frame,
	color_image: np.ndarray,
	image_width: int,
	image_height: int,
	fx: float,
	fy: float,
) -> Tuple[List[List[float]], np.ndarray]:
	"""
	在稳定帧上做精确测量：GrabCut分割 → 凸包 → minAreaRect短边 → 物理直径。
	标注图上只显示 D=X.Xcm。
	"""
	rows: List[List[float]] = []
	annotated = color_image.copy()

	for det_index in range(len(xywh)):
		cx, cy, box_w_px, box_h_px = xywh[det_index].tolist()
		x = int(round(cx)); y = int(round(cy))
		z = _sample_depth_meters(depth_frame, x, y)

		box_x1, box_y1, box_x2, box_y2 = xyxy[det_index].tolist()
		x1 = max(0, int(round(box_x1)))
		y1 = max(0, int(round(box_y1)))
		x2 = min(image_width - 1, int(round(box_x2)))
		y2 = min(image_height - 1, int(round(box_y2)))

		roi = color_image[y1:y2 + 1, x1:x2 + 1]
		short_edge_px, contour, _ = _min_area_rect_short_edge_px(roi) if roi.size else (float("nan"), None, None)
		if not np.isfinite(short_edge_px) or short_edge_px <= 0:
			short_edge_px = float(min(box_w_px, box_h_px))

		physical_diameter_m  = float(short_edge_px * z / ((fx + fy) / 2.0))
		physical_diameter_cm = float(physical_diameter_m * 100.0)
		class_id = float(cls_ids[det_index]) if cls_ids is not None else float("nan")
		conf     = float(confs[det_index])   if confs  is not None else float("nan")

		# 只显示 D
		label = f"D={physical_diameter_cm:.1f}cm"
		_draw_box_label(annotated, x1, y1, x2, y2, label)

		# 画最小外接矩形（青色）
		if contour is not None:
			contour_shifted = contour.copy()
			contour_shifted[:, 0, 0] += x1
			contour_shifted[:, 0, 1] += y1
			rect = cv2.minAreaRect(contour_shifted)
			box_points = np.intp(cv2.boxPoints(rect))
			cv2.drawContours(annotated, [box_points], 0, (255, 255, 0), 2)

		cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)

		rows.append([float(x), float(y), float(z), float(short_edge_px),
					 physical_diameter_m, physical_diameter_cm, float(class_id), float(conf)])

	return rows, annotated


def _get_xyxy_from_result(result, image_width: int, image_height: int):
	"""从 YOLO result 提取归一化修正后的 xyxy / xywh / cls / conf，均为 np.ndarray。"""
	boxes = getattr(result, "boxes", None)
	if boxes is None or len(boxes) == 0:
		return None, None, None, None

	xywh = boxes.xywh.cpu().numpy()
	xyxy = boxes.xyxy.cpu().numpy()
	cls_ids = boxes.cls.cpu().numpy()  if getattr(boxes, "cls",  None) is not None else None
	confs   = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else None

	if xywh.size == 0:
		return None, None, None, None

	if np.max(np.abs(xywh)) <= 1.5:
		xywh = xywh.copy()
		xywh[:, 0] *= image_width;  xywh[:, 2] *= image_width
		xywh[:, 1] *= image_height; xywh[:, 3] *= image_height
	if xyxy.size != 0 and np.max(np.abs(xyxy)) <= 1.5:
		xyxy = xyxy.copy()
		xyxy[:, 0] *= image_width;  xyxy[:, 2] *= image_width
		xyxy[:, 1] *= image_height; xyxy[:, 3] *= image_height

	return xyxy, xywh, cls_ids, confs


def run(args) -> None:
	weights_path = Path(args.weights)
	if not weights_path.exists():
		raise FileNotFoundError(f"未找到模型文件: {weights_path}")

	bag_path = Path(args.bag).expanduser() if args.bag else None
	if bag_path is not None and (not bag_path.exists() or bag_path.suffix.lower() != ".bag"):
		raise FileNotFoundError(f"未找到有效的 bag 文件: {bag_path}")

	_prepare_output_dir(Path(args.output))
	model = YOLO(str(weights_path))
	print(f"[INFO] 模型加载完成: {weights_path}")

	pipeline, profile = _setup_pipeline(bag_path, args.width, args.height, args.fps)
	aligner = rs.align(rs.stream.color)
	window_name = "RealSense YOLO Detection"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	# ── 滑动窗口：缓存最近 STABLE_N 帧的 (color_image, depth_frame, xyxy, xywh, cls, conf) ──
	STABLE_N   = 3        # 连续帧数
	IOU_THRESH = 0.7      # IOU 阈值
	frame_buf: deque = deque(maxlen=STABLE_N)   # 每元素: (color_img, depth_frame, xyxy, xywh, cls, conf)

	xyz_rows: List[List[float]] = []
	measurement_rows: List[List[float]] = []
	meta_rows: List[List[float]] = []
	frame_index = 0

	# 上一次触发稳定测量的帧索引集合（避免同一稳定段重复触发）
	last_stable_frame: int = -STABLE_N

	try:
		while True:
			try:
				frames = pipeline.wait_for_frames(timeout_ms=5000 if bag_path is not None else 1000)
			except RuntimeError as exc:
				if bag_path is not None and (
					"Frame didn't arrive" in str(exc)
					or "device is disconnected" in str(exc)
					or "No frame" in str(exc)
				):
					print("[INFO] bag 读取结束")
					break
				raise

			aligned_frames = aligner.process(frames)
			color_frame = aligned_frames.get_color_frame()
			depth_frame = aligned_frames.get_depth_frame()
			if not color_frame or not depth_frame:
				continue

			color_image      = _frame_to_bgr(color_frame)
			color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
			image_height, image_width = color_image.shape[:2]
			fx = float(color_intrinsics.fx)
			fy = float(color_intrinsics.fy)

			results = model.predict(source=color_image, conf=args.conf, imgsz=args.imgsz, verbose=False)
			result  = results[0] if results else None

			xyxy, xywh, cls_ids, confs = _get_xyxy_from_result(result, image_width, image_height) \
				if result is not None else (None, None, None, None)

			# 把当前帧推入缓冲
			frame_buf.append((color_image.copy(), depth_frame, xyxy, xywh, cls_ids, confs, frame_index))

			# ── 预览帧：YOLO框 + "等待稳定..." 提示，不做精确计算 ──────────────
			preview = color_image.copy()
			stable_triggered = False

			if len(frame_buf) == STABLE_N and (frame_index - last_stable_frame) >= STABLE_N:
				# 检查连续3帧是否稳定：帧0↔帧1、帧1↔帧2 各目标IOU均>0.7
				bufs = list(frame_buf)
				all_stable = True
				for fi in range(STABLE_N - 1):
					xyxy_a = bufs[fi][2]
					xyxy_b = bufs[fi + 1][2]
					if xyxy_a is None or xyxy_b is None:
						all_stable = False
						break
					# 要求两帧检测框数量一致，且每对IOU均满足阈值
					matched = _match_boxes(xyxy_a, xyxy_b, IOU_THRESH)
					if len(matched) != len(xyxy_a) or len(matched) != len(xyxy_b):
						all_stable = False
						break

				if all_stable and xyxy is not None:
					# ── 取中间帧做精确测量 ───────────────────────────────
					mid = bufs[STABLE_N // 2]
					mid_color, mid_depth, mid_xyxy, mid_xywh, mid_cls, mid_conf, mid_fidx = mid

					rows, annotated = _compute_diameters_on_frame(
						mid_xyxy, mid_xywh, mid_cls, mid_conf,
						mid_depth, mid_color,
						image_width, image_height, fx, fy,
					)

					for det_index, row in enumerate(rows):
						x, y, z, min_diameter_px, physical_diameter_m, physical_diameter_cm, class_id, conf = row
						xyz_rows.append([x, y, z])
						measurement_rows.append([x, y, z, min_diameter_px, physical_diameter_m, physical_diameter_cm])
						meta_rows.append([
							float(mid_fidx), float(det_index),
							x, y, z, min_diameter_px,
							physical_diameter_m, physical_diameter_cm,
							class_id, conf,
						])

					preview = annotated
					last_stable_frame = frame_index
					stable_triggered  = True
					print(f"[STABLE] 帧{mid_fidx} 触发稳定测量，检测到 {len(rows)} 个目标")

			if not stable_triggered:
				# 非稳定帧：只画YOLO框，显示"等待稳定"
				if xyxy is not None:
					for i in range(len(xyxy)):
						x1, y1, x2, y2 = [int(round(v)) for v in xyxy[i].tolist()]
						cv2.rectangle(preview, (x1, y1), (x2, y2), (180, 180, 180), 1)
				cv2.putText(preview, "waiting...", (10, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

			cv2.imshow(window_name, preview)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q") or key == 27:
				print("[INFO] 用户退出")
				break

			frame_index += 1

	finally:
		pipeline.stop()
		cv2.destroyAllWindows()

	output_dir = Path(args.output)
	xyz_matrix         = np.asarray(xyz_rows,         dtype=np.float32)
	measurement_matrix = np.asarray(measurement_rows, dtype=np.float32)
	meta_matrix        = np.asarray(meta_rows,        dtype=np.float32)

	np.save(output_dir / "xyz_matrix.npy",          xyz_matrix)
	np.save(output_dir / "measurements_matrix.npy", measurement_matrix)

	csv_path = output_dir / "xyz_records.csv"
	with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
		writer = csv.writer(f)
		writer.writerow(["frame_idx", "det_idx", "x", "y", "z",
						 "min_diameter_px", "physical_diameter_m", "physical_diameter_cm",
						 "class_id", "conf"])
		writer.writerows(meta_matrix.tolist())

	print(f"[INFO] 处理完成，共记录 {len(xyz_rows)} 个稳定目标测量点")
	print(f"[INFO] x/y/z 矩阵:  {output_dir / 'xyz_matrix.npy'}")
	print(f"[INFO] 直径矩阵:    {output_dir / 'measurements_matrix.npy'}")
	print(f"[INFO] 详细记录:    {csv_path}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="RealSense bag/live YOLO depth sampler")
	parser.add_argument("--bag",     type=str,   default="",              help="RealSense .bag 文件路径；留空则打开实时相机")
	parser.add_argument("--weights", type=str,   default=DEFAULT_WEIGHTS, help="YOLO 模型权重路径")
	parser.add_argument("--output",  type=str,   default=DEFAULT_OUTPUT_DIR, help="输出目录")
	parser.add_argument("--conf",    type=float, default=0.4,  help="YOLO 置信度阈值")
	parser.add_argument("--imgsz",   type=int,   default=640,  help="YOLO 推理尺寸")
	parser.add_argument("--width",   type=int,   default=640,  help="实时相机 RGB 宽度")
	parser.add_argument("--height",  type=int,   default=480,  help="实时相机 RGB 高度")
	parser.add_argument("--fps",     type=int,   default=30,   help="实时相机 FPS")
	parser.add_argument("--iou",     type=float, default=0.7,  help="稳定帧 IOU 阈值（默认0.7）")
	return parser


def main() -> None:
	args = build_parser().parse_args()
	run(args)


if __name__ == "__main__":
	main()
