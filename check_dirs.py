import glob, os, csv

dirs = {
  'train_complete':    r'D:\data\pet_output\2000000000\2e9\train\complete_*.npy',
  'train_incomplete':  r'D:\data\pet_output\2000000000\2e9\train\incomplete_*.npy',
  'test_complete':     r'D:\data\pet_output\2000000000\2e9\test\complete_*.npy',
  'test_incomplete':   r'D:\data\pet_output\2000000000\2e9\test\incomplete_*.npy',
  'sinogram_remain':   r'E:\data\pet_output\2000000000\sinogram\reconstructed_*.npy',
  'incomplete_remain': r'D:\data\pet_output\2000000000\listmode_i_6_9_13_16_24_26_32_34\sinogram_incomplete\incomplete_index*_num2000000000.npy',
}
for k, pat in dirs.items():
    files = sorted(glob.glob(pat))
    print(f'{k}: {len(files)}')
    if files:
        print(f'  first: {os.path.basename(files[0])}')
        print(f'  last:  {os.path.basename(files[-1])}')

# Also read CSV to count entries
csv_path = r'D:\data\pet_output\2000000000\2e9\rename_log.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
n_complete   = sum(1 for r in rows if r['type']=='complete')
n_incomplete = sum(1 for r in rows if r['type']=='incomplete')
print(f'\nCSV: {len(rows)} rows  ({n_complete} complete + {n_incomplete} incomplete)')
# Show the first incomplete src to confirm dir name
inc_rows = [r for r in rows if r['type']=='incomplete']
if inc_rows:
    print(f'  first incomplete src: {inc_rows[0]["src"]}')
    print(f'  last  incomplete src: {inc_rows[-1]["src"]}')
