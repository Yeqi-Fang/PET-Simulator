import glob
import hashlib
from collections import defaultdict
from pathlib import Path


def get_md5(filepath, chunk_size=8192):
    """计算文件的 MD5 值"""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def find_duplicates(patterns):
    # 1. 收集所有文件
    all_files = []
    for pattern in patterns:
        matched = glob.glob(pattern)
        print(f"[Pattern] {pattern} => 匹配到 {len(matched)} 个文件")
        all_files.extend(matched)

    all_files = sorted(set(all_files))  # 去重 + 排序
    print(f"\n共收集到 {len(all_files)} 个文件，开始计算 MD5...\n")

    # 2. 计算每个文件的 MD5
    md5_map = defaultdict(list)  # md5 -> [filepath, ...]
    for i, filepath in enumerate(all_files, 1):
        try:
            md5 = get_md5(filepath)
            md5_map[md5].append(filepath)
            print(f"  [{i}/{len(all_files)}] {Path(filepath).name}  =>  {md5}")
        except Exception as e:
            print(f"  [{i}/{len(all_files)}] 读取失败: {filepath}  错误: {e}")

    # 3. 找出重复文件
    duplicates = {md5: paths for md5, paths in md5_map.items() if len(paths) > 1}

    print("\n" + "=" * 60)
    if duplicates:
        print(f"发现 {len(duplicates)} 组重复文件：\n")
        for md5, paths in duplicates.items():
            print(f"  MD5: {md5}")
            for p in paths:
                print(f"    - {p}")
            print()
    else:
        print("未发现任何重复文件（所有文件 MD5 均不同）")
    print("=" * 60)

    return duplicates


if __name__ == "__main__":
    patterns = [
        r"D:\data\pet_output\2000000000\2e9smooth_add\test\complete*.npy",
        r"D:\data\pet_output\2000000000\2e9smooth\test\complete*.npy",
        r"D:\data\pet_output\2000000000\2e9smooth\train\complete*.npy",
    ]

    find_duplicates(patterns)
