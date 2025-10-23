
#################这个是对图片中的每个字母进行一个个分割（外接矩形包起来），然后存储起来######################
##################用于对图片中的字符提取出来，然后自由组合起来############################
import cv2
import numpy as np
import os

# 1. 读取图片
img = cv2.imread(r'D:\PayCard_Detection\paycard_dectection_test1\feng_test\img_6.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 二值化（Otsu自动阈值）
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. 形态学闭运算，填补小孔和去除噪声
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 4. 轮廓检测，只检测外轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5. 过滤掉面积过大的轮廓（防止图片边框被当字符）
height, width = thresh.shape
char_contours = [cnt for cnt in contours if 50 < cv2.contourArea(cnt) < 0.5 * height * width]

# 6. 按x坐标排序
bounding_boxes = [cv2.boundingRect(cnt) for cnt in char_contours]
sorted_boxes = sorted(bounding_boxes, key=lambda b: b[0])

# 7. 创建保存目录
save_dir = 'chars_3'
os.makedirs(save_dir, exist_ok=True)

# 8. 逐个保存字符图片
for idx, (x, y, w, h) in enumerate(sorted_boxes):
    # 添加适当扩展边缘，避免裁切太紧
    pad = 2
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, width)
    y2 = min(y + h + pad, height)
    char_img = img[y1:y2, x1:x2]
    cv2.imwrite(f'{save_dir}/char_{idx+1}.png', char_img)

print(f'分割并保存了 {len(sorted_boxes)} 个字符图片到 {save_dir} 文件夹')

