import os
import re

# train_folder = r"C:\Users\fangy\Desktop\reconstructed\train"

# test_folder = r"C:\Users\fangy\Desktop\reconstructed\test"

# # Specify the output directory or set to None to use a "padded" subfolder

# output_dir = r"C:\Users\fangy\Desktop\reconstructed_"
# # reconstructed_index0_num2000000000 find

# pattern = r"reconstructed_index(\d+)_num2000000000"

# # cnt = 0
# for i in os.listdir(test_folder):
#     if i.endswith(".npy"):
#         # Extract the index from the filename using regex
#         match = re.search(pattern, i)
#         if match:
#             index = match.group(1)
#             print(f"Found index: {index}")
            
#             # new_name = f"reconstructed_from_sinogram_train_incomplete_{int(index) + 1}.npy"
#             # print(f"Renaming {i} to {new_name}")
#             # os.rename(os.path.join(train_folder, i), os.path.join(train_folder, new_name))
#             new_name = f"test_incomplete_{int(index) + 1}.npy"
            
#             os.rename(os.path.join(test_folder, i), os.path.join(output_dir, new_name))
            
base = r"C:\Users\fangy\Desktop\all_prediction_merged_reconstructed\all_prediction_merged_reconstructed_rm_padded_1e-8"

for i in os.listdir(base):
    # print(i)
    # remove the prefix "test_incomplete_"
    new_name = i.replace("reconstructed_from_sinogram_", "")
    
    os.rename(os.path.join(base, i), os.path.join(base, new_name))