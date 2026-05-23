import csv, shutil, os

patch_csv = r"D:\data\pet_output\2000000000\2e9_add\patch_log.csv"
count = 0
errors = 0

with open(patch_csv, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        dst = row['dst']
        src = row['src']
        if os.path.exists(dst):
            os.makedirs(os.path.dirname(src), exist_ok=True)
            shutil.move(dst, src)
            print(f"  [OK] {os.path.basename(dst)} -> {os.path.basename(src)}")
            count += 1
        else:
            print(f"  [SKIP] not found: {dst}")
            errors += 1

print(f"\nDone. moved={count}, skipped={errors}")
