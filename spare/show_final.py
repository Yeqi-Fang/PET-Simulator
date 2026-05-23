import SimpleITK as sitk
import matplotlib.pyplot as plt

# 加载图像
image = sitk.ReadImage(r"D:\Github\PET-Reconstruction\experiments\sr_ffhq_250412_142644\results\0_1_input.img")
img_array = sitk.GetArrayFromImage(image)


print(image.GetSize())  # 打印图像大小
# 显示中间切片
# plt.imshow(img_array[img_array.shape[0]//2], cmap='magma')
# plt.colorbar()
# plt.show()

# convert to numpy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

img_array = sitk.GetArrayFromImage(image)
img_array = img_array.astype(np.float32)
# img_array = img_array[65, :, :]


# print(img_array.shape)  # 打印图像大小

# plt.imshow(img_array, cmap='magma')
# plt.show()

# print max and min

print("max: ", np.max(img_array))
print("min: ", np.min(img_array))
