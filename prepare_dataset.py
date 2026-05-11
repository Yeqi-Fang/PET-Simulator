import os
import shutil
import random
import json

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
SEED = 42
BASE          = "/root/autodl-tmp"
COMPLETE_DIR  = f"{BASE}/osem_complete"
PREDICTED_DIR = f"{BASE}/osem_predicted"

OUTPUT_COMPLETE_TRAIN  = f"{BASE}/complete_train"
OUTPUT_COMPLETE_TEST   = f"{BASE}/complete_test"
OUTPUT_INCOMPLETE_TRAIN = f"{BASE}/incomplete_train"
OUTPUT_INCOMPLETE_TEST  = f"{BASE}/incomplete_test"

SPLIT_LIST_PATH = f"{BASE}/split_list.json"

# ─────────────────────────────────────────────
# Step 1：建立全部 217 个文件的统一命名列表
#   test/1.npy  → test_1.npy
#   train/1.npy → train_1.npy
# ─────────────────────────────────────────────
all_files = []
for i in range(1, 38):          # test: 1~37
    all_files.append(f"test_{i}.npy")
for i in range(1, 181):         # train: 1~180
    all_files.append(f"train_{i}.npy")

assert len(all_files) == 217, "文件数量不对"

# ─────────────────────────────────────────────
# Step 2：随机打乱，分成 180 train / 37 test
# ─────────────────────────────────────────────
random.seed(SEED)
shuffled = all_files.copy()
random.shuffle(shuffled)

split_test  = shuffled[:37]     # 37 个 test
split_train = shuffled[37:]     # 180 个 train

print(f"Train: {len(split_train)} 个, Test: {len(split_test)} 个")

# 保存划分列表（方便复现）
with open(SPLIT_LIST_PATH, "w") as f:
    json.dump({"train": split_train, "test": split_test}, f, indent=2, ensure_ascii=False)
print(f"划分列表已保存到: {SPLIT_LIST_PATH}")

# ─────────────────────────────────────────────
# 辅助函数：根据统一命名反推原始路径
# ─────────────────────────────────────────────
def source_path(base_dir: str, unified_name: str) -> str:
    """
    unified_name: e.g. "test_5.npy" or "train_42.npy"
    → base_dir/test/5.npy  or  base_dir/train/42.npy
    """
    prefix, rest = unified_name.split("_", 1)   # "test", "5.npy"
    return os.path.join(base_dir, prefix, rest)

# ─────────────────────────────────────────────
# Step 3：创建输出目录
# ─────────────────────────────────────────────
for d in [OUTPUT_COMPLETE_TRAIN, OUTPUT_COMPLETE_TEST,
          OUTPUT_INCOMPLETE_TRAIN, OUTPUT_INCOMPLETE_TEST]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# Step 4：按照相同划分列表复制两套数据
# ─────────────────────────────────────────────
def copy_files(file_list, src_base, dst_dir, label):
    ok, missing = 0, []
    for fname in file_list:
        src = source_path(src_base, fname)
        dst = os.path.join(dst_dir, fname)
        if not os.path.exists(src):
            missing.append(src)
            continue
        shutil.copy2(src, dst)
        ok += 1
    print(f"[{label}] 复制 {ok}/{len(file_list)} 个文件 → {dst_dir}")
    if missing:
        print(f"  ⚠️  缺失 {len(missing)} 个源文件:")
        for p in missing:
            print(f"     {p}")

# osem_complete → complete_train / complete_test
copy_files(split_train, COMPLETE_DIR,  OUTPUT_COMPLETE_TRAIN,   "complete  → train")
copy_files(split_test,  COMPLETE_DIR,  OUTPUT_COMPLETE_TEST,    "complete  → test ")

# osem_predicted → incomplete_train / incomplete_test
copy_files(split_train, PREDICTED_DIR, OUTPUT_INCOMPLETE_TRAIN, "predicted → train")
copy_files(split_test,  PREDICTED_DIR, OUTPUT_INCOMPLETE_TEST,  "predicted → test ")

print("\n✅ 全部完成！目录结构：")
for d in [OUTPUT_COMPLETE_TRAIN, OUTPUT_COMPLETE_TEST,
          OUTPUT_INCOMPLETE_TRAIN, OUTPUT_INCOMPLETE_TEST]:
    count = len(os.listdir(d))
    print(f"  {d}  ({count} 个文件)")