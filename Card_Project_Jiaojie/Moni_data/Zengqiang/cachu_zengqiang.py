
import os
import numpy as np
from PIL import Image
import cv2
import random

def random_cutout_on_char(img, min_cut=0.02, max_cut=0.05, holes=2, fill_value=0):
    """
    在黑底白字图片的字符区域内随机擦除若干小块（用黑色填充）。
    img: PIL灰度图像
    min_cut, max_cut: 擦除块面积占字符面积的比例范围
    holes: 每个字符区域擦除块的数量
    fill_value: 擦除块填充值，黑底白字应为0（黑色）
    """
    arr = np.array(img)
    H, W = arr.shape[:2]
    # 直接用THRESH_BINARY，轮廓就是白色字符区域
    _, binary = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        for _ in range(holes):
            cut_ratio = random.uniform(min_cut, max_cut)
            cut_area = int(w * h * cut_ratio)
            cut_w = int(np.sqrt(cut_area))
            cut_h = cut_w
            if cut_w < 1 or cut_h < 1:
                continue
            cut_x = random.randint(x, max(x, x + w - cut_w))
            cut_y = random.randint(y, max(y, y + h - cut_h))
            arr[cut_y:cut_y+cut_h, cut_x:cut_x+cut_w] = fill_value
    return Image.fromarray(arr)

# 批量处理部分
input_dir = r"D:\PayCard_Detection\new_all_data\letters\heidi_baizi\val\val_data_1"
output_dir = r"D:\PayCard_Detection\new_all_data\letters\heidi_baizi\val\val_data_1_cachu" # 输出文件夹

os.makedirs(output_dir, exist_ok=True)

# 支持的图片扩展名
img_exts = ['.png', '.jpg', '.jpeg', '.bmp']

for fname in os.listdir(input_dir):
    if not any(fname.lower().endswith(ext) for ext in img_exts):
        continue
    img_path = os.path.join(input_dir, fname)
    img = Image.open(img_path).convert("L")
    aug_img = random_cutout_on_char(img, min_cut=0.01, max_cut=0.05, holes=2, fill_value=0)
    # 可加后缀区分
    out_path = os.path.join(output_dir, fname)
    aug_img.save(out_path)
    print(f"Processed: {fname}")

print("批量增强完成！")
