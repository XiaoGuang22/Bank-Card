
#####################这个代码是进行字符之间的自由拼接，为了扩充数据集###########################
####################这个是在分割完字符之后，然后设定参数进行随机组合，形成新的数据#########################

import cv2
import numpy as np
import os
import random

# 路径参数
letters_dir = r'D:\PayCard_Detection\paycard_dectection_test1\feng_test\chars_1'  # 26字母图片文件夹

bg_path = r'D:\PayCard_Detection\paycard_dectection_test1\Moni_data\background.png'     # 背景图
save_dir = r'D:\PayCard_Detection\paycard_dectection_test1\feng_test\new_data\val_chars_1'
os.makedirs(save_dir, exist_ok=True)




# 自动裁剪函数：去除多余黑边
def auto_crop(img, thresh=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]
    else:
        return img



# 获取全部字母图片的文件名和字母名
letter_files = sorted([f for f in os.listdir(letters_dir) if f.endswith('.png')])
alphabet = [os.path.splitext(f)[0] for f in letter_files]

num_images = 2000  # 需要生成的图片数量

for img_idx in range(num_images):
    # 随机选5~12个不同字母
    num_letters = random.randint(5, 15)
    #这个选取的字符不可以重复
    # selected_indices = random.sample(range(26), num_letters)
    selected_indices = random.choices(range(26), k=num_letters)
    # selected_indices = random.choices(range(11), k=num_letters)


    selected_letters = [alphabet[i] for i in selected_indices]
    selected_files = [letter_files[i] for i in selected_indices]

    # 随机生成间距
    spaces = [random.randint(10,70) for _ in range(num_letters - 1)]

    # 读取所有字母图片并自动裁剪，记录尺寸
    letter_imgs = []
    heights, widths = [], []
    for f in selected_files:
        img = cv2.imread(os.path.join(letters_dir, f))
        img = auto_crop(img)
        letter_imgs.append(img)
        heights.append(img.shape[0])
        widths.append(img.shape[1])

    max_height = max(heights)
    total_width = sum(widths) + sum(spaces)

    # 计算背景尺寸
    bg_height = max_height + 2
    bg_width = total_width + 2

    # 读取并resize背景图
    bg_img = cv2.imread(bg_path)
    bg_img = cv2.resize(bg_img, (bg_width, bg_height))

    # 拼接字母（字母垂直居中，边缘羽化+阈值处理）
    x = 1  # 左边距
    for i, img in enumerate(letter_imgs):
        h, w = img.shape[:2]
        y = (bg_height - h) // 2

        # 计算alpha蒙版
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alpha = gray.astype(np.float32) / 255.0
        # 腐蚀去杂边
        kernel = np.ones((3,3), np.uint8)
        alpha = cv2.erode(alpha, kernel, iterations=1)
        # 高斯模糊羽化
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        # 阈值处理，极低alpha直接透明
        alpha[alpha < 0.05] = 0
        alpha = np.expand_dims(alpha, axis=2)

        roi = bg_img[y:y+h, x:x+w]
        # 混合
        bg_img[y:y+h, x:x+w] = (img.astype(np.float32) * alpha + roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        x += w
        if i < len(spaces):
            x += spaces[i]

    # 保存图片
    combined_str = ''.join(selected_letters)
    save_path = os.path.join(save_dir, f'{combined_str}.png')
    cv2.imwrite(save_path, bg_img)
    print(f'[{img_idx+1}/{num_images}] 已保存: {save_path}')
