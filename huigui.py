import numpy as np

# 你测量的样本：D_measured（脚本输出cm） vs D_true（卡尺测量cm）
D_true = np.array([2.6, 1.9, 2.2, 2.5, 1.4, 1.6, 2.3])
D_measured     = np.array([3.0, 2.2, 2.7, 2.7, 2.0, 2.0, 2.5])

# 最小二乘线性回归：D_true = a * D_measured + b
A = np.vstack([D_measured, np.ones(len(D_measured))]).T
(a, b), _, _, _ = np.linalg.lstsq(A, D_true, rcond=None)
print(f"校正公式: D_true = {a:.4f} * D_measured + {b:.4f}cm")

# 验证
D_corrected = a * D_measured + b
residuals = D_corrected - D_true
print(f"校正后 MAE = {np.abs(residuals).mean():.3f} cm")