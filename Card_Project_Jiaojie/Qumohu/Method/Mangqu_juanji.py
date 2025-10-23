########################这个代码运用了一种去模糊方法：盲去卷积的方法##################
########################这种方法有个缺点，如果可以知道模糊核的话，那么可以很好去模糊##############
#######################现有的改良方法需要进行一个迭代计算模糊核，消耗时间特别长，不可取##############

# import cv2
# import numpy as np
# from skimage.restoration import richardson_lucy
# from skimage import img_as_float
#
# def blind_deconvolution(blurred_image, num_iterations=10):
#     # 归一化图像
#     blurred_image = img_as_float(blurred_image)
#     # 初始化模糊核
#     estimated_kernel = np.ones((5, 5)) / 25
#
#     current_image = blurred_image
#     for i in range(num_iterations):
#         # 使用Richardson-Lucy算法去卷积
#         restored = richardson_lucy(current_image, estimated_kernel, iterations=10)
#         # 更新模糊核（这里可以用更复杂的方法，但简单起见用随机核）
#         estimated_kernel = np.random.rand(5, 5)
#         estimated_kernel /= estimated_kernel.sum()  # 保持归一化
#
#         current_image = restored
#
#     return restored
#
# # 应用盲去卷积
# blurred_image = cv2.imread(r'D:\PayCard_Detection\paycard_dectection_test1\Muban\img_1.png', 0)
# restored_image_blind = blind_deconvolution(blurred_image)
# cv2.imwrite("Mangqu.png", (restored_image_blind * 255).astype(np.uint8))



##########################盲区卷积的代码，不过这个是指定了模糊核############################
import cv2
import numpy as np
from skimage.restoration import richardson_lucy
from skimage import img_as_float

blurred_image = cv2.imread(r'D:\PayCard_Detection\paycard_dectection_test1\Muban\img_1.png', 0)
blurred_image = img_as_float(blurred_image)
psf = np.ones((5, 5)) / 25  # 简单均匀核

restored = richardson_lucy(blurred_image, psf, iterations=30)
cv2.imwrite("restored_mangqu.png", (restored * 255).astype(np.uint8))

