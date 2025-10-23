######################这个是另外一种去模糊的方法：维纳滤波去模糊方法##################
#####################与盲去卷积方法差不多，也是需要一个准确的模糊核###################


########################这里是维纳滤波去模糊算法########################
# import numpy as np
# import cv2
# from scipy.signal import convolve2d
#
# def wiener_filter(blurred, kernel, noise_var, estimated_noise_var):
#     kernel_ft = np.fft.fft2(kernel, s=blurred.shape)
#     blurred_ft = np.fft.fft2(blurred)
#
#     ratio = np.conj(kernel_ft) / (np.abs(kernel_ft)**2 + estimated_noise_var / noise_var)
#     restored_ft = blurred_ft * ratio
#     restored = np.fft.ifft2(restored_ft)
#
#     return np.abs(restored)
#
# # 读取模糊图像
# blurred_image = cv2.imread(r'D:\PayCard_Detection\paycard_dectection_test1\Muban\img_1.png', 0)
#
# # 定义模糊核（这里使用一个简单的平均模糊核）
# kernel = np.ones((5, 5)) / 25
#
# # 估计噪声方差
# noise_variance = 0.01
# estimated_noise_variance = 0.01
#
# # 去模糊处理
# restored_image = wiener_filter(blurred_image, kernel, noise_variance, estimated_noise_variance)
#
# cv2.imwrite('restored_image.jpg', restored_image)



##################改进后的维纳滤波算法###############################
import numpy as np
import cv2

def wiener_filter(blurred, kernel, noise_var, estimated_noise_var):
    # 计算模糊核和图像的傅里叶变换
    kernel_ft = np.fft.fft2(kernel, s=blurred.shape)
    blurred_ft = np.fft.fft2(blurred)

    # Wiener滤波公式
    ratio = np.conj(kernel_ft) / (np.abs(kernel_ft)**2 + estimated_noise_var / noise_var)
    restored_ft = blurred_ft * ratio
    restored = np.fft.ifft2(restored_ft)

    # 取实部并归一化到0-255
    restored = np.abs(restored)
    restored = (restored - restored.min()) / (restored.max() - restored.min())
    restored = (restored * 255).astype(np.uint8)
    return restored

# 读取模糊图像
blurred_image = cv2.imread(r'D:\PayCard_Detection\paycard_dectection_test1\Qumohu\example_images\img_2.png', 0)

# 定义模糊核
kernel = np.ones((5, 5)) / 25

# 自动估算图像方差
noise_var = np.var(blurred_image)

# 设置不同的噪声估计比例，方便对比
ratios = [0.001, 0.01, 0.05,0.1,0.5,1.0]
for ratio in ratios:
    estimated_noise_var = ratio * noise_var
    restored_image = wiener_filter(blurred_image, kernel, noise_var, estimated_noise_var)
    save_path = f'restored_{ratio}_N.png'
    cv2.imwrite(save_path, restored_image)
    print(f"已保存 {save_path}")

