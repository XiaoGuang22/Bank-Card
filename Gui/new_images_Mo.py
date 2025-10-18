##################这一个是模拟相关数据的，不过模拟的数据都是字体规范的数据############################
import random
import os
from PIL import Image, ImageDraw, ImageFont

# 保存文件夹路径（请修改为你的目标文件夹）
save_dir = "D:\\PayCard_Detection\\Moni_data\\test_images_no"
os.makedirs(save_dir, exist_ok=True)

# 图像尺寸
width, height = 1600, 1200

# 字体路径（请确认字体文件存在）
font_path = "D:\\PayCard_Detection\\Data\\ziti\\Farrington-7B\\Farrington-7B.ttf"
font_large = ImageFont.truetype(font_path, 120)
font_medium = ImageFont.truetype(font_path, 80)
font_small = ImageFont.truetype(font_path, 70)

# 随机生成卡号（4组4位数字）
def random_card_number():
    nums = [str(random.randint(1000, 9999)) for _ in range(4)]
    return ' '.join(nums)

# 随机有效期（MMYY，无分隔符）
def random_expiry():
    mm = str(random.randint(1, 12)).zfill(2)
    yy = str(random.randint(25, 35))
    return f"{mm}{yy}"

# 随机编号（1位数字）
def random_id():
    return str(random.randint(0, 9))

# 随机“姓名”（三组，每组2~4位数字）
def random_numeric_name():
    group1 = ''.join([str(random.randint(0, 9)) for _ in range(3)])
    group2 = ''.join([str(random.randint(0, 9)) for _ in range(2)])
    group3 = ''.join([str(random.randint(0, 9)) for _ in range(4)])
    return f"{group1} {group2} {group3}"

for i in range(20):
    # 创建黑色背景图片
    img = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(img)

    # 白色条带
    bar_width = 40
    draw.rectangle([0, 0, bar_width, height], fill='white')
    draw.rectangle([width-bar_width, 0, width, height], fill='white')

    # 生成内容
    card_number = random_card_number()
    expiry = random_expiry()
    id_number = random_id()
    name = random_numeric_name()

    # 卡号
    draw.text((bar_width + 60, 280), card_number, font=font_large, fill='white', spacing=20)
    # 有效期与编号
    draw.text((bar_width + 700, 500), f"{expiry} {id_number}", font=font_medium, fill='white')
    # “姓名”全部为数字
    draw.text((bar_width + 60, 800), name, font=font_small, fill='white', spacing=20)

    # 保存图片
    img.save(os.path.join(save_dir, f"all_numeric_card_{i+1:02d}.png"))
