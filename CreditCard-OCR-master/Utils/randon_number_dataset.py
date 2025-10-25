"""
银行卡号码数据集生成工具

该脚本用于生成银行卡号码的合成图像数据集，主要用于OCR模型的训练。
功能包括：
1. 生成各种格式的随机银行卡号码
2. 将号码渲染为图像
3. 应用数据增强技术（噪声、擦除等）
4. 批量生成训练数据集

作者：AI Assistant
日期：2024
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import random
import os
import numpy as np

# 字体路径配置
FONT_PATH = "D:\\PayCard_Detection\\Data\\ziti\\Farrington-7B\\Farrington-7B.ttf"
# 备选字体路径（已注释）
# "D:\PayCard_Detection\Data\ziti\ocr-b-maisfontes.18c2\ocr-b.otf"
# FONT_PATH = "D:\\PayCard_Detection\\Data\\ziti\\Times New\\74370-main\\Times New  Roman.TTF\\TIMESBD.TTF"
FONT_SIZE = 50  # 字体大小

def group_numbers(numbers, group_sizes):
    """
    将数字字符串按照指定的大小分组
    
    Args:
        numbers: 数字字符串
        group_sizes: 每组的大小列表，如[4,4,4,4]表示分成4组，每组4个数字
    
    Returns:
        分组后的字符串，用空格分隔
    """
    grouped = []
    idx = 0
    for size in group_sizes:
        grouped.append(numbers[idx:idx+size])
        idx += size
        if idx >= len(numbers):
            break
    # 如果还有剩余数字，添加到最后一组
    if idx < len(numbers):
        grouped.append(numbers[idx:])
    return ' '.join(grouped)

def add_random_noise(img, amount=0.02):
    """
    添加椒盐噪声到图像中，用于数据增强
    
    Args:
        img: PIL图像对象
        amount: 噪声比例，默认0.02（2%的像素点会被噪声替换）
    
    Returns:
        添加噪声后的PIL图像对象
    """
    arr = np.array(img)
    # 计算盐噪声和胡椒噪声的数量
    num_salt = np.ceil(amount * arr.size * 0.5)
    num_pepper = np.ceil(amount * arr.size * 0.5)
    
    # 添加盐噪声（白色像素点）
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in arr.shape]
    arr[tuple(coords)] = 255
    
    # 添加胡椒噪声（黑色像素点）
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in arr.shape]
    arr[tuple(coords)] = 0
    return Image.fromarray(arr)



def random_erase(img, num_rect=1, max_size_ratio=0.2):
    """
    随机擦除图像的一小块区域，模拟数字缺失或污损
    
    Args:
        img: PIL图像对象
        num_rect: 要擦除的矩形区域数量，默认1
        max_size_ratio: 擦除区域的最大尺寸比例，默认0.2（最大为图像尺寸的20%）
    
    Returns:
        擦除后的PIL图像对象
    """
    arr = np.array(img)
    h, w = arr.shape
    for _ in range(num_rect):
        # 随机确定擦除区域的宽度和高度
        erase_w = random.randint(3, int(w * max_size_ratio))
        erase_h = random.randint(3, int(h * max_size_ratio))
        # 随机确定擦除区域的位置
        x = random.randint(0, w - erase_w)
        y = random.randint(0, h - erase_h)
        # 将选定区域设置为黑色（擦除）
        arr[y:y+erase_h, x:x+erase_w] = 0
    return Image.fromarray(arr)

def generate_card_number_image(numbers, save_path):
    """
    生成银行卡号码图像并保存
    
    Args:
        numbers: 银行卡号码字符串
        save_path: 保存路径
    """
    # 计算图像尺寸
    char_width = FONT_SIZE
    img_width = char_width * len(numbers) + 40  # 每个字符宽度 + 边距
    img_height = int(FONT_SIZE * 1.7)  # 字符高度的1.7倍
    
    # 创建黑色背景图像
    img = Image.new('L', (img_width, img_height), color=0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    
    # 第一次绘制文字
    for i, char in enumerate(numbers):
        x = 10 + i * char_width
        y = 5
        draw.text((x, y), char, fill=255, font=font)
    
    # 重新创建绘制对象并再次绘制文字（增强效果）
    draw = ImageDraw.Draw(img)
    for i, char in enumerate(numbers):
        x = 10 + i * char_width
        y = 5
        draw.text((x, y), char, fill=255, font=font)
    
    # 应用图像滤波器增强效果
    img = img.filter(ImageFilter.MinFilter(1))  # 最小值滤波
    img = img.filter(ImageFilter.MaxFilter(3))  # 最大值滤波（膨胀效果）

    # --- 数据增强部分 ---
    # 随机添加椒盐噪声（50%概率）
    if random.random() < 0.5:
        img = add_random_noise(img, amount=random.uniform(0.01, 0.03))
    
    # 随机仿射变形（已注释，可启用）
    # if random.random() < 0.4:
    #     img = random_affine(img)
    
    # 随机擦除部分区域，模拟数字缺失（50%概率）
    if random.random() > 0.5:
        img = random_erase(img, num_rect=random.randint(1,2), max_size_ratio=0.15)

    # 保存图像
    img.save(save_path)

def random_card_number(group_sizes):
    """
    生成随机银行卡号码
    
    Args:
        group_sizes: 每组数字的大小列表，如[4,4,4,4]表示4组，每组4个数字
    
    Returns:
        格式化的银行卡号码字符串，用空格分隔各组
    """
    nums = []
    for size in group_sizes:
        # 为每组生成指定数量的随机数字
        nums.append(''.join(str(random.randint(0, 9)) for _ in range(size)))
    return ' '.join(nums)

def generate_dataset(num_images=10, save_dir='dataset'):
    """
    生成银行卡号码数据集
    
    Args:
        num_images: 要生成的图像数量，默认10张
        save_dir: 保存目录，默认'dataset'
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_images):
        # 定义不同的银行卡号码分组模式
        group_patterns = [
            [4,4,4,4],  # 标准16位卡号：4-4-4-4
            [4,6,5],     # 15位卡号：4-6-5
            [3,4,4,5],   # 16位卡号：3-4-4-5
            [4,5,6],     # 15位卡号：4-5-6
            [5,5,5]      # 15位卡号：5-5-5
        ]
        
        # 随机选择一种分组模式
        group_sizes = random.choice(group_patterns)
        
        # 生成随机银行卡号码
        card_number = random_card_number(group_sizes)
        
        # 生成文件名（将空格替换为下划线）
        img_name = card_number.replace(' ', '_') + '.png'
        save_path = os.path.join(save_dir, img_name)
        
        # 生成并保存图像
        generate_card_number_image(card_number, save_path)

# 生成10张测试图片到指定目录
generate_dataset(10, "D:\\PayCard_Detection\\Data\\test7")