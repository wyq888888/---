#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RealSense D435 bag/live YOLO 实时预览
用法：
  实时相机：python preview.py --weights model.pt
  bag文件： python preview.py --weights model.pt --bag xxx.bag
按 q 或 ESC 退出
"""

import argparse
import ctypes
import os
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


def _to_short_path(path: Path) -> Path:
    """Windows下将含非ASCII字符的路径转为短路径，避免RealSense SDK报错。"""
    if os.name != "nt" or not path.exists():
        return path
    if path.as_posix().isascii():
        return path
    try:
        buf = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, len(buf)):
            return Path(buf.value)
    except Exception:
        pass
    return path


def run(args):
    # ── 加载模型 ──────────────────────────────────────────────────
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"模型文件不存在: {weights}")
    model = YOLO(str(weights))
    print(f"[INFO] 模型加载完成: {weights}")

    # ── 配置 RealSense pipeline ───────────────────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()
    bag_path = Path(args.bag).expanduser() if args.bag else None

    if bag_path:
        bag_path = _to_short_path(bag_path)
        if not bag_path.exists():
            raise FileNotFoundError(f"bag文件不存在: {bag_path}")
        config.enable_device_from_file(str(bag_path), repeat_playback=args.loop)
        print(f"[INFO] 读取bag: {bag_path}  loop={args.loop}")
    else:
        config.enable_stream(rs.stream.color, args.width, args.height,
                             rs.format.bgr8, args.fps)
        print(f"[INFO] 实时相机 {args.width}x{args.height} @ {args.fps}fps")

    profile = pipeline.start(config)

    if bag_path:
        playback = profile.get_device().as_playback()
        playback.set_real_time(True)   # 按原速播放，改False则尽快读完

    # ── 窗口 ──────────────────────────────────────────────────────
    win = "YOLO Preview  |  q / ESC 退出"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_idx = 0
    try:
        while True:
            # 读帧
            try:
                timeout = 5000 if bag_path else 1000
                frames  = pipeline.wait_for_frames(timeout_ms=timeout)
            except RuntimeError as e:
                if bag_path:
                    print("[INFO] bag读取结束")
                    break
                raise

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # numpy BGR图像
            img = np.asanyarray(color_frame.get_data())
            if color_frame.profile.format() == rs.format.rgb8:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # ── YOLO推理 ──────────────────────────────────────────
            results = model.predict(
                source=img,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
                show=False,
            )

            # 用YOLO自带的annotate绘制框/mask/标签，省去手写绘制逻辑
            annotated = results[0].plot() if results else img.copy()

            # 左上角显示帧号和置信度阈值
            cv2.putText(annotated, f"frame={frame_idx}  conf>={args.conf}",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(win, annotated)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("[INFO] 用户退出")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f"[INFO] 共处理 {frame_idx} 帧")


def main():
    parser = argparse.ArgumentParser(description="RealSense + YOLO 实时预览")
    parser.add_argument("--weights", type=str, required=True,   help="YOLO .pt 模型路径")
    parser.add_argument("--bag",     type=str, default="",      help="bag文件路径；留空则打开实时相机")
    parser.add_argument("--conf",    type=float, default=0.4,   help="置信度阈值（默认0.4）")
    parser.add_argument("--imgsz",   type=int,   default=640,   help="推理尺寸（默认640）")
    parser.add_argument("--loop",    action="store_true",       help="bag循环播放")
    parser.add_argument("--width",   type=int,   default=640,   help="实时相机宽度")
    parser.add_argument("--height",  type=int,   default=480,   help="实时相机高度")
    parser.add_argument("--fps",     type=int,   default=30,    help="实时相机FPS")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
