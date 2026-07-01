# import torch 
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0) )

import os
import shutil
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# print(os.path.exists(r"C:\text5\dataset\data.yaml"))
from ultralytics import YOLO

def main():
    # 1. 选择模型（最小模型，最稳）
    model = YOLO('yolov8n-obb.pt')

    # 2. 开始训练
    model.train(
        data=r"C:\text5\tape\tape.yaml",  # data.yaml 的绝对路径
        epochs=60,                          # 初次训练推荐 20~30
        imgsz=640,                          # 输入尺寸
        batch=4,                            # 显存不够就改 4 或 2
        device=0,                           # 0 = 第一张 GPU；没有 GPU 可改为 "cpu"
        workers=4,                          # Windows 下 2~4 比较稳
        project="runs/tape",              # 输出目录
        name="train_exp",                   # 实验名
        exist_ok=True                       # 允许覆盖同名实验
    )

    # 3. 训练完成后，打印模型路径
    source_best = Path("runs/tape/train_exp/weights/tape.pt")
    target_best = Path("tape.pt")
    if source_best.exists():
        shutil.copy2(source_best, target_best)

    print("Training finished!")
    print("Best model saved at:")
    print(str(target_best))


if __name__ == "__main__":
    main()
