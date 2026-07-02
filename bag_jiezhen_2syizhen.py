"""
RealSense D435 .bag 文件 RGB/Depth 帧提取工具
功能：按 2 秒间隔读取 .bag 文件，提取 RGB 帧和对应深度数据，按时间戳命名
依赖：pyrealsense2, opencv-python, numpy
"""


import argparse
import ctypes
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs


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


def _prepare_output_dir(output_dir: Path) -> Tuple[Path, Path]:
	rgb_dir = output_dir / "rgb"
	depth_dir = output_dir / "depth"
	rgb_dir.mkdir(parents=True, exist_ok=True)
	depth_dir.mkdir(parents=True, exist_ok=True)
	return rgb_dir, depth_dir


def _frame_to_bgr(color_frame) -> np.ndarray:
	color_image = np.asanyarray(color_frame.get_data())
	color_format = color_frame.get_profile().as_video_stream_profile().format()

	if color_format == rs.format.rgb8:
		return cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
	if color_format == rs.format.bgr8:
		return color_image
	if color_image.ndim == 3 and color_image.shape[2] == 3:
		return cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
	return color_image


def _save_image_unicode(path: Path, image: np.ndarray) -> None:
	path = Path(path)
	extension = path.suffix.lower()
	if extension not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
		raise ValueError(f"不支持的图片格式: {path}")

	success, encoded_image = cv2.imencode(extension, image)
	if not success:
		raise RuntimeError(f"图片编码失败: {path}")

	path.parent.mkdir(parents=True, exist_ok=True)
	encoded_image.tofile(str(path))


def _save_color_and_depth_frames(bag_path: Path, output_dir: Path, interval_seconds: float = 2.0) -> None:
	original_path = bag_path
	bag_path = _to_windows_short_path(bag_path)
	output_dir = Path(output_dir)
	rgb_dir, depth_dir = _prepare_output_dir(output_dir)

	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_device_from_file(str(bag_path), repeat_playback=False)

	profile = pipeline.start(config)
	device = profile.get_device()
	playback = device.as_playback()
	playback.set_real_time(False)

	align = rs.align(rs.stream.color)

	print(f"正在读取 bag 文件: {original_path}")
	if bag_path != original_path:
		print(f"已转换为 Windows 短路径: {bag_path}")
	print(f"输出目录: {output_dir}")
	print(f"保存间隔: {interval_seconds:.1f} 秒")

	saved_count = 0
	last_saved_ms: Optional[float] = None
	first_timestamp_ms: Optional[float] = None

	try:
		while True:
			try:
				frames = pipeline.wait_for_frames(timeout_ms=1000)
			except RuntimeError:
				break

			aligned_frames = align.process(frames)
			color_frame = aligned_frames.get_color_frame()
			depth_frame = aligned_frames.get_depth_frame()

			if not color_frame or not depth_frame:
				continue

			timestamp_ms = float(color_frame.get_timestamp())
			if first_timestamp_ms is None:
				first_timestamp_ms = timestamp_ms
				last_saved_ms = timestamp_ms - interval_seconds * 1000.0

			if last_saved_ms is not None and (timestamp_ms - last_saved_ms) < interval_seconds * 1000.0:
				continue

			color_image_bgr = _frame_to_bgr(color_frame)
			depth_image = np.asanyarray(depth_frame.get_data())

			time_ms = int(round(timestamp_ms - first_timestamp_ms))
			file_stem = f"frame_{saved_count:05d}_t{time_ms:07d}ms"

			color_path = rgb_dir / f"{file_stem}.png"
			depth_raw_path = depth_dir / f"{file_stem}.npy"

			_save_image_unicode(color_path, color_image_bgr)
			np.save(depth_raw_path, depth_image)

			saved_count += 1
			last_saved_ms = timestamp_ms
			print(f"已保存第 {saved_count} 组: {file_stem}")

		print(f"完成，共保存 {saved_count} 组图片。")
	finally:
		pipeline.stop()


def main() -> None:
	parser = argparse.ArgumentParser(description="按 2 秒间隔从 RealSense D435 .bag 文件中提取 RGB/Depth 帧")
	parser.add_argument(
		"--bag",
		nargs="?",
		default="",
		help=".bag 文件路径，例如: snapshots/test.bag",
	)
	parser.add_argument(
		"--output",
		default=r"C:\text5\2026_6_4_bag\clip_data",
		help="输出文件夹",
	)
	parser.add_argument(
		"--interval",
		type=float,
		default=2.0,
		help="抽帧间隔，单位秒，默认 2 秒",
	)
	args = parser.parse_args()

	bag_path = Path(args.bag).expanduser()
	if not args.bag:
		raise ValueError("请提供 .bag 文件路径，例如: python bag_jiezhen_2syizhen.py your_file.bag")

	if not bag_path.exists() or bag_path.suffix.lower() != ".bag":
		raise FileNotFoundError(f"未找到有效的 .bag 文件: {bag_path}")

	_save_color_and_depth_frames(bag_path, Path(args.output), interval_seconds=args.interval)


if __name__ == "__main__":
	main()
