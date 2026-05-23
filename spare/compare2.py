import skimage

path = r"C:\Users\fangy\Desktop\converted_mat_files_rm_1e-8_rotated\test\test_26_55.mat"

# Load the .mat file
import scipy.io as sio
mat_data = sio.loadmat(path)
# print(mat_data.keys())

img = mat_data['img']


# display

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.imshow(img[0, :, :].T, cmap='magma')
plt.colorbar()
plt.show()
