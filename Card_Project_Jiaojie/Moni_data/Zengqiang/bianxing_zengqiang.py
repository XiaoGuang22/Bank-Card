#####################批量处理：包括（变形，旋转，高斯）适用于白底黑字############################

import cv2
import numpy as np
import random
import os
from scipy.ndimage import map_coordinates, gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    """弹性形变（Elastic deformation）"""
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1, 1), (x + dx).reshape(-1, 1)

    distorted = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    return distorted.astype(np.uint8)

def pad_image(img, pad=40, value=0):
    '''在四周加黑色边框，防止旋转后字符被裁剪'''
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)

def random_rotate(img, angle_range=2):
    '''随机旋转，保证所有内容都显示出来'''
    h, w = img.shape
    angle = random.uniform(-angle_range, angle_range)
    # 计算旋转后的新尺寸
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    # 调整旋转中心
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated = cv2.warpAffine(img, M, (new_w, new_h), borderValue=0)
    return rotated

def add_gaussian_noise(img, mean=0, sigma=10):
    noise = np.random.normal(mean, sigma, img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def crop_to_content(img):
    '''自动裁剪回最小外接矩形，适用于黑底白字'''
    # 只需将THRESH_BINARY_INV换为THRESH_BINARY
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]
    else:
        return img

def process_image(img_path, out_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return
    # 1. 弹性形变
    aug_img = elastic_transform(img, alpha=18, sigma=6)
    # 2. 加padding（黑色）
    aug_img = pad_image(aug_img, pad=40)
    # 3. 随机旋转
    aug_img = random_rotate(aug_img, angle_range=1)
    # 4. 自动裁剪
    aug_img = crop_to_content(aug_img)
    # 5. 加高斯噪声
    aug_img = add_gaussian_noise(aug_img, sigma=20)
    # 6. 保存
    cv2.imwrite(out_path, aug_img)

def batch_augment_images(input_dir, output_dir, suffix=""):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_dir, filename)
            # 生成输出文件名
            name, ext = os.path.splitext(filename)
            out_filename = f"{name}{suffix}{ext}"
            out_path = os.path.join(output_dir, out_filename)
            process_image(img_path, out_path)
            print(f"已处理: {filename} -> {out_filename}")

if __name__ == "__main__":
    input_dir = r"D:\PayCard_Detection\new_all_data\letters\heidi_baizi\val\val_data_6"
    output_dir = r"D:\PayCard_Detection\new_all_data\letters\heidi_baizi\val\val_data_6_bianxing"
    batch_augment_images(input_dir, output_dir)
