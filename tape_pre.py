"""tape OBB 目标初步识别。

输入 RGB 图和对应深度图，使用 `tape.pt` 做 YOLOv8-OBB 推理，
输出每个目标的四个角点，以及下边/左边对应的两点。
同时根据检测框中心点采样深度执行过滤处理（内部使用，不在输出层返回过滤图）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


DEFAULT_WEIGHTS = Path(__file__).with_name("tape.pt")
DEPTH_TOLERANCE_M = 2 # 深度过滤容差，单位：米



def _load_image(image_or_path: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image_or_path, np.ndarray):
        return image_or_path.copy()

    image_path = Path(image_or_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    return image


def _load_depth(depth_or_path: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(depth_or_path, np.ndarray):
        depth = depth_or_path.copy()
    else:
        depth_path = Path(depth_or_path)
        if depth_path.suffix.lower() == ".npy":
            depth = np.load(depth_path)
        else:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"无法读取深度文件: {depth_path}")

    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def _load_model(weights_path: str | Path) -> YOLO:
    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"模型文件不存在: {weights}")
    return YOLO(str(weights))


def _normalize_polygons(polygons: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    if polygons.size == 0:
        return polygons

    normalized = polygons.astype(np.float32, copy=True)
    if np.max(np.abs(normalized)) <= 1.5:
        normalized[:, :, 0] *= image_width
        normalized[:, :, 1] *= image_height
    return normalized


def _order_points_clockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def _extract_polygons(result: Any, image_width: int, image_height: int) -> np.ndarray:
    obb = getattr(result, "obb", None)
    if obb is None or len(obb) == 0:
        return np.empty((0, 4, 2), dtype=np.float32)

    polygons = None

    if hasattr(obb, "xyxyxyxy") and obb.xyxyxyxy is not None:
        polygons = obb.xyxyxyxy.cpu().numpy()
    elif hasattr(obb, "xywhr") and obb.xywhr is not None:
        try:
            from ultralytics.utils.ops import xywhr2xyxyxyxy

            polygons = xywhr2xyxyxyxy(obb.xywhr.cpu().numpy())
        except Exception:
            polygons = None

    if polygons is None:
        return np.empty((0, 4, 2), dtype=np.float32)

    polygons = np.asarray(polygons, dtype=np.float32)
    if polygons.ndim == 2 and polygons.shape[-1] == 8:
        polygons = polygons.reshape(-1, 4, 2)
    if polygons.ndim != 3 or polygons.shape[1:] != (4, 2):
        return np.empty((0, 4, 2), dtype=np.float32)

    return _normalize_polygons(polygons, image_width, image_height)


def _polygon_center(points: np.ndarray) -> tuple[int, int]:
    center_x = int(round(float(np.mean(points[:, 0]))))
    center_y = int(round(float(np.mean(points[:, 1]))))
    return center_x, center_y


def _depth_to_meters(depth_map: np.ndarray) -> np.ndarray:
    depth = depth_map.astype(np.float32, copy=True)
    if np.issubdtype(depth_map.dtype, np.integer):
        depth /= 1000.0
        return depth

    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size > 0 and float(np.max(valid)) > 20.0:
        depth /= 1000.0
    return depth


def _ensure_depth_shape(depth_map: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    if depth_map.shape[:2] == (target_height, target_width):
        return depth_map
    return cv2.resize(depth_map, (target_width, target_height), interpolation=cv2.INTER_NEAREST)


def _sample_depth_meters(depth_src, x: int, y: int, radius: int = 2) -> float:
    """从深度源采样有效深度，返回单位：米。"""
    if isinstance(depth_src, rs.depth_frame):
        width = depth_src.get_width()
        height = depth_src.get_height()
        for r in range(radius + 1):
            x_min = max(0, x - r)
            x_max = min(width - 1, x + r)
            y_min = max(0, y - r)
            y_max = min(height - 1, y + r)
            values = []
            for yy in range(y_min, y_max + 1):
                for xx in range(x_min, x_max + 1):
                    z = float(depth_src.get_distance(int(xx), int(yy)))
                    if z > 0:
                        values.append(z)
            if values:
                return float(np.median(values))
        return float("nan")

    depth_map = depth_src
    h, w = depth_map.shape[:2]
    for r in range(radius + 1):
        x_min = max(0, x - r)
        x_max = min(w - 1, x + r)
        y_min = max(0, y - r)
        y_max = min(h - 1, y + r)
        region = depth_map[y_min:y_max + 1, x_min:x_max + 1].astype(np.float32)
        valid = region[(region > 0) & np.isfinite(region)]
        if valid.size > 0:
            return float(np.median(valid))
    return float("nan")


def _filter_rgb_by_depth(rgb_image: np.ndarray, depth_map_m: np.ndarray, center_x: int, center_y: int, tolerance_m: float = DEPTH_TOLERANCE_M) -> tuple[np.ndarray, float]:
    center_depth_m = _sample_depth_meters(depth_map_m, center_x, center_y, radius=2)
    if not np.isfinite(center_depth_m) or center_depth_m <= 0:
        return rgb_image.copy(), center_depth_m

    valid_mask = np.isfinite(depth_map_m) & (depth_map_m > 0)
    keep_mask = valid_mask & (np.abs(depth_map_m - center_depth_m) <= tolerance_m)

    filtered_rgb = np.zeros_like(rgb_image)
    filtered_rgb[keep_mask] = rgb_image[keep_mask]
    return filtered_rgb, center_depth_m


def _pick_edge(points: np.ndarray, mode: str) -> dict[str, list[list[float]]]:
    ordered = _order_points_clockwise(points)
    edges = [
        (ordered[i], ordered[(i + 1) % 4])
        for i in range(4)
    ]

    if mode == "bottom":
        edge_index = max(range(4), key=lambda idx: ((edges[idx][0][1] + edges[idx][1][1]) / 2.0,
                                                    (edges[idx][0][0] + edges[idx][1][0]) / 2.0))
    elif mode == "left":
        edge_index = min(range(4), key=lambda idx: ((edges[idx][0][0] + edges[idx][1][0]) / 2.0,
                                                    (edges[idx][0][1] + edges[idx][1][1]) / 2.0))
    else:
        raise ValueError(f"不支持的边类型: {mode}")

    edge_points = edges[edge_index]
    return {
        "edge_index": edge_index,
        "points": [[float(edge_points[0][0]), float(edge_points[0][1])],
                    [float(edge_points[1][0]), float(edge_points[1][1])]],
    }


def detect_tape_edges(
    image_or_path: str | Path | np.ndarray,
    depth_or_path: str | Path | np.ndarray | None = None,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    conf: float = 0.25,
    imgsz: int = 640,
) -> dict[str, Any]:
    """返回 tape 目标的下边/左边两点信息。"""
    image = _load_image(image_or_path)
    depth_map_m = None
    if depth_or_path is not None and str(depth_or_path) != "":
        depth_map = _load_depth(depth_or_path)
        depth_map = _ensure_depth_shape(depth_map, image.shape[0], image.shape[1])
        depth_map_m = _depth_to_meters(depth_map)
    model = _load_model(weights_path)

    results = model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False, show=False)
    if not results:
        return {
            "detections": [],
            "selected_detection": None,
        }

    result = results[0]
    image_height, image_width = image.shape[:2]
    polygons = _extract_polygons(result, image_width, image_height)

    cls_values = None
    conf_values = None
    obb = getattr(result, "obb", None)
    if obb is not None:
        if getattr(obb, "cls", None) is not None:
            cls_values = obb.cls.cpu().numpy()
        if getattr(obb, "conf", None) is not None:
            conf_values = obb.conf.cpu().numpy()

    detections: list[dict[str, Any]] = []
    for index, polygon in enumerate(polygons):
        bottom_edge = _pick_edge(polygon, "bottom")
        left_edge = _pick_edge(polygon, "left")
        polygon_points = [[float(point[0]), float(point[1])] for point in polygon]

        detections.append({
            "index": index,
            "class_id": int(cls_values[index]) if cls_values is not None and index < len(cls_values) else None,
            "conf": float(conf_values[index]) if conf_values is not None and index < len(conf_values) else None,
            "points": polygon_points,
            "bottom_edge": bottom_edge,
            "left_edge": left_edge,
        })

    selected_detection = None
    if detections:
        selected_detection = detections[0]
        selected_polygon = polygons[0]
        if depth_map_m is not None:
            center_x, center_y = _polygon_center(selected_polygon)
            _, center_depth_m = _filter_rgb_by_depth(image, depth_map_m, center_x, center_y, tolerance_m=DEPTH_TOLERANCE_M)
            selected_detection["center"] = [center_x, center_y]
            selected_detection["center_depth_m"] = center_depth_m
            selected_detection["depth_tolerance_m"] = DEPTH_TOLERANCE_M

    return {
        "detections": detections,
        "selected_detection": selected_detection,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="tape OBB 目标初步识别")
    parser.add_argument("image", help="RGB图片路径")
    parser.add_argument("depth", help="深度文件路径（npy 或 16位深度图）")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="tape.pt 权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    args = parser.parse_args()

    result = detect_tape_edges(args.image, args.depth, weights_path=args.weights, conf=args.conf, imgsz=args.imgsz)
    printable = {
        "detections": result["detections"],
        "selected_detection": result["selected_detection"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()