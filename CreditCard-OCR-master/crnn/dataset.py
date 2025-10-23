"""
CRNN 银行卡号识别数据集定义

提供训练/验证/预测三种模式的数据读取与预处理：
- 从文件名提取标签文本
- 图像转灰度、缩放到固定尺寸、归一化到[-1, 1]
- 将可变长标签打包为(targets, target_lengths)
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CardDataset(Dataset):
    """银行卡号OCR数据集。

    约定：
    - 训练/验证模式下，`image_dir` 为包含图片的目录；文件名需包含标签文本，形如:
      "<label>-xxx.jpg" 或 "<label>.png"，其中 label 允许包含字符'/'（用作分隔符）。
    - 预测模式下，`image_dir` 直接传入 `PIL.Image` 对象。

    标签编码：
    - CTC 需要保留 0 做空白符，因此字符集从 1 开始编码。
    """
    # CHARS = '0123456789'
    CHARS = '0123456789/'
    # 将字符映射为从1开始的标签（0保留为CTC空白）
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    # 反向映射，便于解码可视化
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, image_dir, mode, img_height, img_width):
        """初始化数据集。

        Args:
            image_dir: 训练/验证为目录路径；预测为 PIL.Image 对象
            mode: "train" | "val" | "pred"
            img_height: 预处理后图像高度
            img_width: 预处理后图像宽度
        """
        texts = []
        self.mode = mode
        self.image_dir = image_dir
        self.img_height = img_height
        self.img_width = img_width
        # 训练/验证模式下，从文件名中解析标签
        if mode == "train" or mode == "val":
            file_names, texts = self._load_from_raw_files()
            self.file_names = file_names
        self.texts = texts

    def _load_from_raw_files(self):
        """从目录扫描文件并从文件名提取标签文本。

        提取规则：去除扩展名后，按 '-' 与空格取首段；将下划线'_'替换为'/'。
        例如："6222_8888-1.jpg" -> "6222/8888"
        """
        file_names = []
        texts = []
        file_names = os.listdir(self.image_dir)
        for file_name in file_names:
            # 取文件名主体，优先 '-' 与空格前部分
            text = file_name.split('.')[0].split('-')[0].split(' ')[0]
            # 将 '_' 标记替换为 '/'，与训练字符集保持一致
            # texts.append(text.replace('_', ''))
            texts.append(text.replace('_', '/'))

        return file_names, texts

    def __len__(self):
        """返回数据集大小。

        预测模式只返回单样本（传入的那张图）。
        """
        if self.mode == "pred":
            return 1
        else:
            return len(self.file_names)
        
    def __getitem__(self, index):
        """读取并预处理一个样本。

        Returns:
            - 训练/验证: (image, targets, target_lengths)
            - 预测: image
        """
        if self.mode == "train" or self.mode == "val":
            file_name = self.file_names[index]
            file_path = os.path.join(self.image_dir,file_name)
            image = Image.open(file_path)
        elif self.mode == "pred":
            # 此时image_dir为PIL.Image对象
            image = self.image_dir

        # 图像预处理：转灰度 -> 缩放到固定大小 -> 转为NCHW -> 归一化到[-1, 1]
        image = image.convert('L').resize((self.img_width, self.img_height))
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        # 训练/验证返回图像与目标标签；预测仅返回图像
        if len(self.texts) != 0:
            text = self.texts[index]
            # 将字符序列映射为整数标签序列（从1开始）
            target = [self.CHAR2LABEL[c] for c in text]
            # CTC 需要提供每个样本的目标长度
            target_length = [len(target)]
            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image

# 不定长序列的批处理：
# - 图像尺寸一致，可直接 stack
# - 标签为1D不定长序列，需按batch拼接为一条长向量，并保留每条序列长度
def cardnumber_collate_fn(batch):
    """自定义 collate_fn 以适配 CTC 的(targets, target_lengths)格式。"""
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
