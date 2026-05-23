import numpy as np
from scipy.ndimage import rotate
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False



def rotate_around_first_axis(image, angle, fill_value=1e-8):
    """
    绕第一个轴（axis=0）旋转3D图像
    
    参数:
    - image: 3D numpy数组，输入图像 (128, 128, 128)
    - angle: 旋转角度（度）
    - fill_value: 旋转后空缺区域填充值
    
    返回:
    - 旋转后的图像
    """
    # 在axes=(1,2)平面内旋转，即绕第一个轴旋转
    rotated_image = rotate(image, angle, axes=(1, 2), reshape=False, 
                          order=1, mode='constant', cval=fill_value)
    return rotated_image

def find_optimal_rotation_angle(target_image, source_image, angle_range=(-10, 10), 
                               angle_step=0.1, fill_value=1e-8):
    """
    寻找最佳旋转角度使source_image与target_image对齐
    
    参数:
    - target_image: 目标参考图像
    - source_image: 需要旋转的源图像
    - angle_range: 角度搜索范围 (min_angle, max_angle)
    - angle_step: 角度搜索步长
    - fill_value: 旋转后空缺区域填充值
    
    返回:
    - best_angle: 最佳旋转角度
    - best_score: 最佳角度下的相似度分数
    - scores: 所有测试角度的分数列表
    """
    angles = np.arange(angle_range[0], angle_range[1] + angle_step, angle_step)
    scores = []
    
    best_angle = 0
    best_score = float('-inf')
    
    for angle in angles:
        # 旋转源图像
        rotated = rotate_around_first_axis(source_image, angle, fill_value)
        
        # 计算相似度分数（SSIM）
        score = ssim(target_image, rotated, 
                     data_range=target_image.max() - target_image.min())
        
        scores.append((angle, score))
        
        # 如果当前分数更好，更新最佳分数和角度
        if score > best_score:
            best_score = score
            best_angle = angle
    
    return best_angle, best_score, scores

def plot_similarity_scores(scores, best_angle, best_score):
    """
    绘制相似度分数与旋转角度的关系
    """
    angles, similarity = zip(*scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(angles, similarity, 'b-')
    plt.plot(best_angle, best_score, 'ro', markersize=8)
    plt.axvline(x=best_angle, color='r', linestyle='--', alpha=0.5)
    
    plt.title('图像相似度(SSIM) vs. 旋转角度')
    plt.xlabel('旋转角度（度）')
    plt.ylabel('SSIM值')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 标注最佳角度
    plt.annotate(f'最佳角度: {best_angle:.2f}°\nSSIM: {best_score:.4f}',
                xy=(best_angle, best_score), xytext=(best_angle + 1, best_score),
                arrowprops=dict(arrowstyle='->'))
    
    plt.savefig("rotation_angle_optimization.pdf")
    # plt.show()

def visualize_alignment(target_image, source_image, best_angle, slice_index=64, fill_value=1e-8):
    """
    可视化旋转前后的对齐效果
    """
    # 使用最佳角度旋转源图像
    rotated_image = rotate_around_first_axis(source_image, best_angle, fill_value)
    
    # 创建三个子图
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # 显示目标图像切片
    im0 = axs[0].imshow(target_image[slice_index, :, :], cmap='magma', interpolation='nearest')
    axs[0].set_title(f"目标图像 (切片 {slice_index})")
    axs[0].axis('off')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    # 显示旋转前的源图像切片
    im1 = axs[1].imshow(source_image[slice_index, :, :], cmap='magma', interpolation='nearest')
    axs[1].set_title(f"旋转前的源图像 (切片 {slice_index})")
    axs[1].axis('off')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    
    # 显示旋转后的源图像切片
    im2 = axs[2].imshow(rotated_image[slice_index, :, :], cmap='magma', interpolation='nearest')
    axs[2].set_title(f"旋转后的源图像 (切片 {slice_index})\n角度: {best_angle:.2f}°")
    axs[2].axis('off')
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("rotation_alignment_visualization.pdf")
    plt.show()

def find_best_angle(target_path, source_path, fill_value=1e-8):
    """
    主函数：查找最佳旋转角度并可视化结果
    """
   
    # 加载图像
    target_image = np.load(target_path)
    source_image = np.load(source_path)
    
    print("目标图像形状:", target_image.shape)
    print("源图像形状:", source_image.shape)
    
    # 根据您的代码，可能需要翻转源图像
    # 您可以根据需要注释或取消这行
    flipped_source = np.flip(source_image, axis=2)
    
    # 查找最佳旋转角度
    print("\n正在搜索最佳旋转角度...")
    best_angle, best_score, scores = find_optimal_rotation_angle(
        target_image, flipped_source,  # 或使用source_image如果不需要翻转
        angle_range=(-8, -3),           # 角度搜索范围
        angle_step=0.1,                # 角度步长
        fill_value=1e-8                # 空缺填充值
    )
    
    print(f"最佳旋转角度: {best_angle:.4f} 度")
    print(f"最佳角度下的SSIM: {best_score:.4f}")
    
    # 绘制相似度分数
    plot_similarity_scores(scores, best_angle, best_score)
    
    # 可视化最佳角度下的对齐效果（使用与您代码中相同的切片索引71）
    # visualize_alignment(target_image, flipped_source, best_angle, slice_index=71)
    
    # return best_angle, best_score
    rotated_image = rotate_around_first_axis(flipped_source, best_angle, fill_value)
    # flip 
    rotated_image = np.flip(rotated_image, axis=2)
    return rotated_image
    
    # except Exception as e:
    #     print(f"错误: {e}")

if __name__ == "__main__":
    import os
    
    # 设置要处理的图像索引
    # i = 7
    # 定义图像路径
    # target_path = rf"C:\Users\fangy\Desktop\reconstructed_rm_padded_1e-8\test_incomplete_{i}.npy"  # 目标图像路径
    # source_path = rf"C:\Users\fangy\Desktop\all_prediction_merged_reconstructed\all_prediction_merged_reconstructed_padded\test_incomplete_{i}.npy"  # 源图像路径
    # find_best_angle(target_path, source_path)
    
    input_dir = r"C:\Users\fangy\Desktop\all_prediction_merged_reconstructed\all_prediction_merged_reconstructed_rm_padded_1e-8"
    target_dir = r"C:\Users\fangy\Desktop\reconstructed_rm_padded_1e-8"
    output_dir = r"C:\Users\fangy\Desktop\all_prediction_merged_reconstructed\all_prediction_merged_reconstructed_rm_padded_1e-8_rotated"
    
    for i in os.listdir(input_dir):
        if i.endswith(".npy"):
            target_path = os.path.join(target_dir, i)
            source_path = os.path.join(input_dir, i)
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 定义输出路径
            output_path = os.path.join(output_dir, i)
            
            # 检查输出文件是否已存在
            if os.path.exists(output_path):
                print(f"输出文件已存在，跳过: {output_path}")
                continue
            
            # 调用函数
            rotated_image = find_best_angle(target_path, source_path)
            
            # 保存旋转后的图像
            np.save(output_path, rotated_image)