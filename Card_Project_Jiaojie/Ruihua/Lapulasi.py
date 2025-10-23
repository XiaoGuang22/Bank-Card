# import cv2
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
#
# # 读取图像
# img = cv2.imread(r'D:\PayCard_Detection\paycard_dectection_test1\Ruihua\_.png')
#
# # 定义锐化卷积核(以拉普拉斯算子为例)
# kernel = np.array([[-1,-1,-1],
#                    [-1, 9,-1],
#                    [-1,-1,-1]])
#
# # 应用锐化卷积核
# sharp_img = cv2.filter2D(img, -1, kernel)
#
# # 显示原图和锐化后的图像
# # cv2.imshow('Original Image', img)
# # cv2.imshow('Sharpened Image', sharp_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# cv2.imwrite("lapulasi.png",sharp_img)

#####################一种图像边缘的增强方法：拉普拉斯锐化方法###########################
import cv2
import numpy as np

# 1. 读取图像
img = cv2.imread(r'D:\PayCard_Detection\paycard_dectection_test1\Ruihua\_.png')

# 2. 转为灰度图（可选，彩色图像也可以直接处理）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. 应用拉普拉斯算子
laplacian = cv2.Laplacian(gray, cv2.CV_64F)  # 得到浮点型结果
laplacian = np.uint8(np.absolute(laplacian)) # 取绝对值并转为uint8

# 4. 锐化处理
alpha = 1.0  # 锐化强度
sharpened = cv2.addWeighted(gray, 1, laplacian, alpha, 0)

# # 5. 显示结果
# cv2.imshow('Original', gray)
# cv2.imshow('Laplacian', laplacian)
# cv2.imshow('Sharpened', sharpened)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 6. 保存结果（可选）
cv2.imwrite('sharpened.jpg', sharpened)



