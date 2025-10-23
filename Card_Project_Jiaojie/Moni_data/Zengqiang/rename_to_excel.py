######################这一个主要就应用深度学习了，因为现在的深度学习的方法是将图片的名字和相应的标签名映射存储在一个excel表中########
####################所以图片的命名需要从：1.png,2.png,...,n.png,然后excel中有两列，filename和labels


import os
import cv2
import pandas as pd

def rename_and_convert_to_png(input_dir, output_dir, output_excel):
    # 创建目标文件夹
    os.makedirs(output_dir, exist_ok=True)
    # 获取所有图片文件（按文件名排序）
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    files.sort()  # 保证顺序

    records = []
    #这个start不需要减1
    for idx, old_name in enumerate(files, start=40001):
        old_path = os.path.join(input_dir, old_name)
        old_name_no_ext = os.path.splitext(old_name)[0]  # 原文件名不带扩展名
        new_name = f"{idx}.png"  # 新名字带扩展名
        new_path = os.path.join(output_dir, new_name)
        # 读取图片并保存为PNG格式到新文件夹
        img = cv2.imread(old_path)
        if img is not None:
            cv2.imwrite(new_path, img)
            records.append([new_name, old_name_no_ext])
        else:
            print(f"读取失败，跳过：{old_name}")

    # 保存映射关系到Excel
    df = pd.DataFrame(records, columns=['filename', 'labels'])
    df.to_excel(output_excel, index=False)
    print(f"已完成转换、重命名并保存到新文件夹，标签表已存储到：{output_excel}")

if __name__ == "__main__":
    input_dir = r"D:\PayCard_Detection\new_data_no_zeng\train\train_chars_3"     # 原图片文件夹路径
    output_dir = r"D:\PayCard_Detection\new_data_no_zeng\train\rename\train_picture_3"   # 新图片文件夹路径
    output_excel = r"D:\PayCard_Detection\new_data_no_zeng\train\rename\train_picture_3.xlsx"  # Excel输出路径
    rename_and_convert_to_png(input_dir, output_dir, output_excel)



############################兼容中文###############################
# import os
# import cv2
# import numpy as np
# import pandas as pd
#
# def rename_and_convert_to_png(input_dir, output_dir, output_excel):
#     os.makedirs(output_dir, exist_ok=True)
#     files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
#     files.sort()
#     records = []
#     for idx, old_name in enumerate(files, start=1):
#         old_path = os.path.join(input_dir, old_name)
#         old_name_no_ext = os.path.splitext(old_name)[0]
#         new_name = f"{idx}.png"
#         new_path = os.path.join(output_dir, new_name)
#         # 用兼容中文路径的方式读取和保存
#         try:
#             data = np.fromfile(old_path, dtype=np.uint8)
#             img = cv2.imdecode(data, cv2.IMREAD_COLOR)
#             if img is not None:
#                 ok, buf = cv2.imencode('.png', img)
#                 if ok:
#                     buf.tofile(new_path)
#                     records.append([new_name, old_name_no_ext])
#                 else:
#                     print(f"写入失败，跳过：{old_name}")
#             else:
#                 print(f"读取失败，跳过：{old_name}")
#         except Exception as e:
#             print(f"异常，跳过：{old_name}，原因：{e}")
#
#     df = pd.DataFrame(records, columns=['filename', 'labels'])
#     df.to_excel(output_excel, index=False)
#     print(f"已完成转换、重命名并保存到新文件夹，标签表已存储到：{output_excel}")
#
# if __name__ == "__main__":
#     input_dir = r"D:\PayCard_Detection\new_data_no_zeng\train\train_chars_1"
#     output_dir = r"D:\PayCard_Detection\new_data_no_zeng\train\rename\train_picture_1"
#     output_excel = r"D:\PayCard_Detection\new_all_data\YanZhengMa\CRNN\rename\train_picture.xlsx"
#     rename_and_convert_to_png(input_dir, output_dir, output_excel)
