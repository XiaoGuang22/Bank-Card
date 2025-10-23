import os
import json
from PIL import Image

# 1. 加载标注文件
label_path = "D:\\PayCard_Detection\\Moni_data\\test.json"  # 标注文件路径
with open(label_path, 'r', encoding='utf-8') as f:
    label_data = json.load(f)

# 2. 解析标注信息
regions = label_data['regions']
fields_rects = {region['field']: region['rect'] for region in regions}

# 3. 需要处理的图片文件夹
img_dir = "D:\\PayCard_Detection\\Moni_data\\test_images_no"  # 需要裁剪的图片所在文件夹
save_dir = "D:/PayCard_Detection/Moni_data/cropped"    # 裁剪结果保存文件夹
os.makedirs(save_dir, exist_ok=True)

# 4. 批量处理图片
for img_name in os.listdir(img_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path)
    for field, rect in fields_rects.items():
        # rect格式：[x1, y1, x2, y2]
        crop_img = img.crop(rect)
        base_name = os.path.splitext(img_name)[0]
        save_path = os.path.join(save_dir, f"{base_name}_{field}.png")
        crop_img.save(save_path)
