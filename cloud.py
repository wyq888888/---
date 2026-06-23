#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""RealSense D435 bag/live YOLO depth sampler.

使用说明：
- 支持命令--bag从 .bag 文件读取，如果不指定--bag就直接打开 RealSense 实时相机读取
- 使用指定的 YOLO 模型对 RGB 流做目标识别
- 对每个检测框中心点读取对应深度值
- 计算每个目标轮廓的最小外接矩形短边像素长度，并换算物理直径
- 保存 [x, y, z, min_diameter_px, physical_diameter_m, physical_diameter_mm] 矩阵，以及带元数据的 CSV
"""

import argparse
import csv
import ctypes
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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


def _align_frames(frames) -> Tuple[Optional[rs.video_frame], Optional[rs.depth_frame]]:
	aligned = rs.align(rs.stream.color).process(frames)
	return aligned.get_color_frame(), aligned.get_depth_frame()


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


def _draw_box_label(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str) -> None:
	color = (0, 255, 0)
	text_color = (255, 255, 255)
	bg_color = (0, 128, 0)
	cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.5
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
	gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	kernel = np.ones((3, 3), np.uint8)
	best_contour = None
	best_area = 0.0
	best_mask = None

	for thresh_flag in (cv2.THRESH_BINARY + cv2.THRESH_OTSU, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU):
		_, mask = cv2.threshold(blurred, 0, 255, thresh_flag)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if not contours:
			continue
		contour = max(contours, key=cv2.contourArea)
		area = float(cv2.contourArea(contour))
		if area > best_area:
			best_area = area
			best_contour = contour
			best_mask = mask

	if best_contour is None or best_area <= 0:
		return float("nan"), None, None

	rect = cv2.minAreaRect(best_contour)
	short_edge_px = float(min(rect[1]))
	if short_edge_px <= 0:
		return float("nan"), best_contour, best_mask

	return short_edge_px, best_contour, best_mask


def _extract_xy_from_result(
	result,
	depth_frame: rs.depth_frame,
	image_width: int,
	image_height: int,
	intrinsics,
	color_image: np.ndarray,
) -> Tuple[List[List[float]], np.ndarray]:
	rows: List[List[float]] = []
	annotated = color_image.copy()
	boxes = getattr(result, "boxes", None)
	if boxes is None or len(boxes) == 0:
		return rows, annotated

	xywh = boxes.xywh.cpu().numpy()
	xyxy = boxes.xyxy.cpu().numpy()
	cls_ids = boxes.cls.cpu().numpy() if getattr(boxes, "cls", None) is not None else None
	confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else None
	if xywh.size == 0:
		return rows, annotated

	# 兼容某些结果里返回归一化坐标的情况
	if np.max(np.abs(xywh)) <= 1.5:
		xywh = xywh.copy()
		xywh[:, 0] *= float(image_width)
		xywh[:, 1] *= float(image_height)
		xywh[:, 2] *= float(image_width)
		xywh[:, 3] *= float(image_height)
	if xyxy.size != 0 and np.max(np.abs(xyxy)) <= 1.5:
		xyxy = xyxy.copy()
		xyxy[:, 0] *= float(image_width)
		xyxy[:, 1] *= float(image_height)
		xyxy[:, 2] *= float(image_width)
		xyxy[:, 3] *= float(image_height)

	fx = float(intrinsics.fx)
	fy = float(intrinsics.fy)

	for det_index, box in enumerate(xywh):
		cx, cy, box_w_px, box_h_px = box.tolist()
		x = int(round(cx))
		y = int(round(cy))
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
		physical_diameter_m = float(short_edge_px * z / ((fx + fy) / 2.0))
		physical_diameter_mm = float(physical_diameter_m * 1000.0)
		class_id = float(cls_ids[det_index]) if cls_ids is not None else float("nan")
		conf = float(confs[det_index]) if confs is not None else float("nan")
		label = f"d={physical_diameter_mm:.1f}mm | {short_edge_px:.1f}px | z={z:.2f}m"
		_draw_box_label(annotated, x1, y1, x2, y2, label)
		if contour is not None:
			contour_shifted = contour.copy()
			contour_shifted[:, 0, 0] += x1
			contour_shifted[:, 0, 1] += y1
			rect = cv2.minAreaRect(contour_shifted)
			box_points = cv2.boxPoints(rect)
			box_points = np.intp(box_points)
			cv2.drawContours(annotated, [box_points], 0, (255, 255, 0), 2)
		cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
		rows.append([
			float(x),
			float(y),
			float(z),
			float(short_edge_px),
			physical_diameter_m,
			physical_diameter_mm,
			float(class_id),
			float(conf),
		])

	return rows, annotated


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

	xyz_rows: List[List[float]] = []
	measurement_rows: List[List[float]] = []
	meta_rows: List[List[float]] = []
	frame_index = 0

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

			color_image = _frame_to_bgr(color_frame)
			color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
			image_height, image_width = color_image.shape[:2]
			results = model.predict(source=color_image, conf=args.conf, imgsz=args.imgsz, verbose=False)
			result = results[0] if results else None

			if result is not None:
				rows, annotated = _extract_xy_from_result(
					result,
					depth_frame,
					image_width,
					image_height,
					color_intrinsics,
					color_image,
				)
				if annotated is None:
					annotated = color_image.copy()
				for det_index, row in enumerate(rows):
					x, y, z, min_diameter_px, physical_diameter_m, physical_diameter_mm, class_id, conf = row
					xyz_rows.append([x, y, z])
					measurement_rows.append([x, y, z, min_diameter_px, physical_diameter_m, physical_diameter_mm])
					meta_rows.append([
						float(frame_index),
						float(det_index),
						x,
						y,
						z,
						min_diameter_px,
						physical_diameter_m,
						physical_diameter_mm,
						class_id,
						conf,
					])
			else:
				annotated = color_image.copy()

			cv2.imshow(window_name, annotated)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q") or key == 27:
				print("[INFO] 用户退出")
				break

			frame_index += 1

	finally:
		pipeline.stop()
		cv2.destroyAllWindows()

	output_dir = Path(args.output)
	xyz_matrix = np.asarray(xyz_rows, dtype=np.float32)
	measurement_matrix = np.asarray(measurement_rows, dtype=np.float32)
	meta_matrix = np.asarray(meta_rows, dtype=np.float32)

	npy_path = output_dir / "xyz_matrix.npy"
	measurement_npy_path = output_dir / "measurements_matrix.npy"
	csv_path = output_dir / "xyz_records.csv"
	np.save(npy_path, xyz_matrix)
	np.save(measurement_npy_path, measurement_matrix)
	with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
		writer = csv.writer(f)
		writer.writerow([
			"frame_idx",
			"det_idx",
			"x",
			"y",
			"z",
			"min_diameter_px",
			"physical_diameter_m",
			"physical_diameter_mm",
			"class_id",
			"conf",
		])
		writer.writerows(meta_matrix.tolist())

	print(f"[INFO] 处理完成，共记录 {len(xyz_rows)} 个目标中心点")
	print(f"[INFO] x/y/z 矩阵: {npy_path}")
	print(f"[INFO] 直径矩阵: {measurement_npy_path}")
	print(f"[INFO] 详细记录: {csv_path}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="RealSense bag/live YOLO depth sampler")
	parser.add_argument("--bag", type=str, default="", help="RealSense .bag 文件路径；留空则打开实时相机")
	parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="YOLO 模型权重路径")
	parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="输出目录")
	parser.add_argument("--conf", type=float, default=0.4, help="YOLO 置信度阈值")
	parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸")
	parser.add_argument("--width", type=int, default=640, help="实时相机 RGB 宽度")
	parser.add_argument("--height", type=int, default=480, help="实时相机 RGB 高度")
	parser.add_argument("--fps", type=int, default=30, help="实时相机 FPS")
	return parser


def main() -> None:
	args = build_parser().parse_args()
	run(args)


if __name__ == "__main__":
	main()
