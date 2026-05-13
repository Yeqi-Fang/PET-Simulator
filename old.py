import shutil
from pathlib import Path

train_dir = Path(r"D:\PET\dataset\train_npy_crop")
test_dir  = Path(r"D:\PET\dataset\test_npy_crop")
final_dir = Path(r"D:\PET\dataset\npy_crop")

for npy_file in train_dir.glob("*.npy"):
    new_name = npy_file.stem + "_train.npy"
    shutil.copy2(npy_file, final_dir / new_name)  # 原文件保留

for npy_file in test_dir.glob("*.npy"):
    new_name = npy_file.stem + "_test.npy"
    shutil.copy2(npy_file, final_dir / new_name)  # 原文件保留