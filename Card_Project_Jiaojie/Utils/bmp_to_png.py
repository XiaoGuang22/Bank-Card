##########################这个是一个将图片bmp格式转化为png格式的代码################
#######################烫印的保存下来的数据格式是：bmp格式，所以可能后期处理需要进行转换###########
import os
from PIL import Image

# 指定BMP图片所在文件夹
input_folder = 'D:\\字符识别项目\\Data\\data_1_bmp'
output_folder = 'D:\\字符识别项目\\Data\\data_1_jpeg'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有BMP文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.bmp'):
        bmp_path = os.path.join(input_folder, filename)
        jpg_path = os.path.join(output_folder, filename.rsplit('.', 1)[0] + '.jpg')
        img = Image.open(bmp_path)
        img.save(jpg_path, 'JPEG')
        print(f"{filename} 转换完成！")