# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt
# from skimage import io, img_as_float, img_as_ubyte
# from skimage.exposure import histogram
#
# # ---------- Step 1: 去模糊（模糊知识变换） ----------
# def fuzzy_knowledge(intensity):
#     if intensity <= 0.27:
#         drak = 1.0
#     elif intensity >= 0.5:
#         drak = 0.0
#     else:
#         drak = (0.5 - intensity) / 0.22
#
#     if intensity >= 0.72:
#         brig = 1.0
#     elif intensity <= 0.5:
#         brig = 0.0
#     else:
#         brig = (intensity - 0.5) / 0.22
#
#     if intensity >= 0.72 or intensity <= 0.27:
#         gray = 0.0
#     elif intensity <= 0.5:
#         gray = (intensity - 0.27) / 0.22
#     else:
#         gray = (0.72 - intensity) / 0.22
#
#     return drak, gray, brig
#
# def fuzzy_image_transform(image_path, save_dir):
#     f = io.imread(image_path)
#     f = img_as_float(f)
#     if f.ndim == 3:
#         f = f.mean(axis=2)
#     M, N = f.shape
#     g = np.zeros((M, N))
#
#     for x in range(M):
#         for y in range(N):
#             drak, gray, brig = fuzzy_knowledge(f[x, y])
#             g[x, y] = (drak * 0 + gray * 0.5 + brig * 1) / (drak + gray + brig)
#
#     # 保存去模糊结果
#     fuzzy_path = os.path.join(save_dir, 'step1_fuzzy.png')
#     io.imsave(fuzzy_path, img_as_ubyte(g))
#
#     # 保存原图和处理结果对比图
#     plt.figure(figsize=(10,5))
#     plt.subplot(1,2,1)
#     plt.imshow(f, cmap='gray', vmin=0, vmax=1)
#     plt.title('Original Image')
#     plt.subplot(1,2,2)
#     plt.imshow(g, cmap='gray', vmin=0, vmax=1)
#     plt.title('Fuzzy Result')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'step1_compare.png'))
#     plt.close()
#
#     # 保存直方图
#     plt.figure(figsize=(10,5))
#     plt.subplot(1,2,1)
#     h, bins = histogram(f)
#     h = h / (M*N)
#     plt.bar(bins, h, width=1/255)
#     plt.title('Histogram: Original')
#     plt.subplot(1,2,2)
#     h, bins = histogram(g)
#     h = h / (M*N)
#     plt.bar(bins, h, width=1/255)
#     plt.title('Histogram: Fuzzy')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'step1_hist.png'))
#     plt.close()
#
#     return fuzzy_path
#
# # ---------- Step 2: 高频增强 ----------
# def manualShiftDFT(complexI):
#     M, N = complexI.shape[:2]
#     complexI_shifted = np.zeros_like(complexI)
#     for i in range(M):
#         for j in range(N):
#             newRow = i + M//2 if i < M//2 else i - M//2
#             newCol = j + N//2 if j < N//2 else j - N//2
#             complexI_shifted[newRow, newCol] = complexI[i, j]
#     return complexI_shifted
#
# def lvboqi(M, N, D0=40):
#     H = np.zeros((M, N), np.float64)
#     for i in range(M):
#         for j in range(N):
#             D = np.sqrt((i - M/2)**2 + (j - N/2)**2)
#             H[i, j] = np.exp(-(D**2) / (2 * (D0**2)))
#     return H
#
# def gaotonglvbo(g_shifted, one_minus_h):
#     result = np.zeros_like(g_shifted)
#     for i in range(g_shifted.shape[0]):
#         for j in range(g_shifted.shape[1]):
#             h_val = one_minus_h[i, j]
#             result[i, j, 0] = g_shifted[i, j, 0] * h_val
#             result[i, j, 1] = g_shifted[i, j, 1] * h_val
#     return result
#
# def gaotong(f):
#     m, n = f.shape
#     M1 = cv2.getOptimalDFTSize(m)
#     N1 = cv2.getOptimalDFTSize(n)
#     padded = cv2.copyMakeBorder(f, 0, M1-m, 0, N1-n, cv2.BORDER_CONSTANT, value=0)
#
#     I = np.float64(padded)
#     g = cv2.dft(I, flags=cv2.DFT_COMPLEX_OUTPUT)
#     g_shifted = manualShiftDFT(g)
#     M, N = g_shifted.shape[:2]
#     H = lvboqi(M, N)
#     F = 0.5 + 0.75 * (1 - H)
#     result_highpass2 = gaotonglvbo(g_shifted, F)
#     G_shifted = manualShiftDFT(result_highpass2)
#     J2 = cv2.idft(G_shifted, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
#     J3 = J2[:m, :n]
#     J4 = np.abs(J3)
#     J4 = np.clip(J4, 0, 255).astype(np.uint8)
#     return J4
#
# def highpass_process(fuzzy_path, save_dir):
#     img = cv2.imread(fuzzy_path, 0)
#     result = gaotong(img)
#     highpass_path = os.path.join(save_dir, 'step2_highpass.png')
#     cv2.imwrite(highpass_path, result)
#
#     # 保存对比图
#     plt.figure(figsize=(10,5))
#     plt.subplot(1,2,1)
#     plt.imshow(img, cmap='gray')
#     plt.title('Fuzzy Input')
#     plt.subplot(1,2,2)
#     plt.imshow(result, cmap='gray')
#     plt.title('Highpass Result')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'step2_compare.png'))
#     plt.close()
#
#     return highpass_path
#
# # ---------- Step 3: 二值化+连通域分析 ----------
# def bin_and_connect(highpass_path, save_dir):
#     img = cv2.imread(highpass_path, 0)
#     _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     bin_path = os.path.join(save_dir, 'step3_binary.png')
#     cv2.imwrite(bin_path, binary)
#
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
#     filtered = np.zeros_like(binary)
#     for i in range(1, num_labels):
#         area = stats[i, cv2.CC_STAT_AREA]
#         if area > 100:
#             filtered[labels == i] = 255
#
#     filtered_path = os.path.join(save_dir, 'step3_filtered.png')
#     cv2.imwrite(filtered_path, filtered)
#
#     # 保存对比图
#     plt.figure(figsize=(10,5))
#     plt.subplot(1,2,1)
#     plt.imshow(binary, cmap='gray')
#     plt.title('Binary')
#     plt.subplot(1,2,2)
#     plt.imshow(filtered, cmap='gray')
#     plt.title('Filtered')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'step3_compare.png'))
#     plt.close()
#
#     return filtered_path
#
# # ---------- 主流程 ----------
# def main(image_path, save_dir):
#     if not os.path.exists(image_path):
#         print(f"Error: File {image_path} does not exist.")
#         return
#
#     os.makedirs(save_dir, exist_ok=True)
#     # Step 1: 去模糊
#     fuzzy_path = fuzzy_image_transform(image_path, save_dir)
#     print(f"Step 1 Done: {fuzzy_path}")
#
#     # Step 2: 高频增强
#     highpass_path = highpass_process(fuzzy_path, save_dir)
#     print(f"Step 2 Done: {highpass_path}")
#
#     # Step 3: 二值化+连通域分析
#     filtered_path = bin_and_connect(highpass_path, save_dir)
#     print(f"Step 3 Done: {filtered_path}")
#
# # ---------- 使用方法 ----------
# if __name__ == "__main__":
#     # 修改为你的图片路径和保存目录
#     image_path = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\Letter\test_images\img_28.png"
#     save_dir = r"D:\PayCard_Detection\paycard_dectection_test1\Qumohu\process_results_method_1\test_debug_images_6"
#     main(image_path, save_dir)



##############下面的这个代码，加入了开运算的形态学处理###########################
# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt
# from skimage import io, img_as_float, img_as_ubyte
# from skimage.exposure import histogram
#
# # ---------- Step 1: 去模糊（模糊知识变换） ----------
# def fuzzy_knowledge(intensity):
#     if intensity <= 0.27:
#         drak = 1.0
#     elif intensity >= 0.5:
#         drak = 0.0
#     else:
#         drak = (0.5 - intensity) / 0.22
#
#     if intensity >= 0.72:
#         brig = 1.0
#     elif intensity <= 0.5:
#         brig = 0.0
#     else:
#         brig = (intensity - 0.5) / 0.22
#
#     if intensity >= 0.72 or intensity <= 0.27:
#         gray = 0.0
#     elif intensity <= 0.5:
#         gray = (intensity - 0.27) / 0.22
#     else:
#         gray = (0.72 - intensity) / 0.22
#
#     return drak, gray, brig
#
# def fuzzy_image_transform(image_path, save_dir):
#     f = io.imread(image_path)
#     f = img_as_float(f)
#     if f.ndim == 3:
#         f = f.mean(axis=2)
#     M, N = f.shape
#     g = np.zeros((M, N))
#
#     for x in range(M):
#         for y in range(N):
#             drak, gray, brig = fuzzy_knowledge(f[x, y])
#             g[x, y] = (drak * 0 + gray * 0.5 + brig * 1) / (drak + gray + brig)
#
#     # 保存去模糊结果
#     fuzzy_path = os.path.join(save_dir, 'step1_fuzzy.png')
#     io.imsave(fuzzy_path, img_as_ubyte(g))
#
#     # 保存原图和处理结果对比图
#     plt.figure(figsize=(10,5))
#     plt.subplot(1,2,1)
#     plt.imshow(f, cmap='gray', vmin=0, vmax=1)
#     plt.title('Original Image')
#     plt.subplot(1,2,2)
#     plt.imshow(g, cmap='gray', vmin=0, vmax=1)
#     plt.title('Fuzzy Result')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'step1_compare.png'))
#     plt.close()
#
#     # 保存直方图
#     plt.figure(figsize=(10,5))
#     plt.subplot(1,2,1)
#     h, bins = histogram(f)
#     h = h / (M*N)
#     plt.bar(bins, h, width=1/255)
#     plt.title('Histogram: Original')
#     plt.subplot(1,2,2)
#     h, bins = histogram(g)
#     h = h / (M*N)
#     plt.bar(bins, h, width=1/255)
#     plt.title('Histogram: Fuzzy')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'step1_hist.png'))
#     plt.close()
#
#     return fuzzy_path
#
# # ---------- Step 2: 高频增强 ----------
# def manualShiftDFT(complexI):
#     M, N = complexI.shape[:2]
#     complexI_shifted = np.zeros_like(complexI)
#     for i in range(M):
#         for j in range(N):
#             newRow = i + M//2 if i < M//2 else i - M//2
#             newCol = j + N//2 if j < N//2 else j - N//2
#             complexI_shifted[newRow, newCol] = complexI[i, j]
#     return complexI_shifted
#
# def lvboqi(M, N, D0=40):
#     H = np.zeros((M, N), np.float64)
#     for i in range(M):
#         for j in range(N):
#             D = np.sqrt((i - M/2)**2 + (j - N/2)**2)
#             H[i, j] = np.exp(-(D**2) / (2 * (D0**2)))
#     return H
#
# def gaotonglvbo(g_shifted, one_minus_h):
#     result = np.zeros_like(g_shifted)
#     for i in range(g_shifted.shape[0]):
#         for j in range(g_shifted.shape[1]):
#             h_val = one_minus_h[i, j]
#             result[i, j, 0] = g_shifted[i, j, 0] * h_val
#             result[i, j, 1] = g_shifted[i, j, 1] * h_val
#     return result
#
# def gaotong(f):
#     m, n = f.shape
#     M1 = cv2.getOptimalDFTSize(m)
#     N1 = cv2.getOptimalDFTSize(n)
#     padded = cv2.copyMakeBorder(f, 0, M1-m, 0, N1-n, cv2.BORDER_CONSTANT, value=0)
#
#     I = np.float64(padded)
#     g = cv2.dft(I, flags=cv2.DFT_COMPLEX_OUTPUT)
#     g_shifted = manualShiftDFT(g)
#     M, N = g_shifted.shape[:2]
#     H = lvboqi(M, N)
#     F = 0.5 + 0.75 * (1 - H)
#     result_highpass2 = gaotonglvbo(g_shifted, F)
#     G_shifted = manualShiftDFT(result_highpass2)
#     J2 = cv2.idft(G_shifted, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
#     J3 = J2[:m, :n]
#     J4 = np.abs(J3)
#     J4 = np.clip(J4, 0, 255).astype(np.uint8)
#     return J4
#
# def highpass_process(fuzzy_path, save_dir):
#     img = cv2.imread(fuzzy_path, 0)
#     result = gaotong(img)
#     highpass_path = os.path.join(save_dir, 'step2_highpass.png')
#     cv2.imwrite(highpass_path, result)
#
#     # 保存对比图
#     plt.figure(figsize=(10,5))
#     plt.subplot(1,2,1)
#     plt.imshow(img, cmap='gray')
#     plt.title('Fuzzy Input')
#     plt.subplot(1,2,2)
#     plt.imshow(result, cmap='gray')
#     plt.title('Highpass Result')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'step2_compare.png'))
#     plt.close()
#
#     return highpass_path
#
# # ---------- Step 3: 二值化+中值滤波+形态学开运算+连通域分析 ----------
# def bin_and_connect(highpass_path, save_dir):
#     img = cv2.imread(highpass_path, 0)
#     img_blur = cv2.medianBlur(img, 3)  # 先中值滤波
#     _, binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # 形态学开运算去毛刺
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
#
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)
#     filtered = np.zeros_like(binary_clean)
#     for i in range(1, num_labels):
#         area = stats[i, cv2.CC_STAT_AREA]
#         if area > 100:
#             filtered[labels == i] = 255
#
#     cv2.imwrite(os.path.join(save_dir, 'step3_binary.png'), binary)
#     cv2.imwrite(os.path.join(save_dir, 'step3_binary_clean.png'), binary_clean)
#     cv2.imwrite(os.path.join(save_dir, 'step3_filtered.png'), filtered)
#
#     # 可视化对比
#     plt.figure(figsize=(15,5))
#     plt.subplot(1,3,1)
#     plt.imshow(binary, cmap='gray'); plt.title('Binary')
#     plt.subplot(1,3,2)
#     plt.imshow(binary_clean, cmap='gray'); plt.title('Morph Open')
#     plt.subplot(1,3,3)
#     plt.imshow(filtered, cmap='gray'); plt.title('Filtered')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'step3_compare.png'))
#     plt.close()
#
#     return os.path.join(save_dir, 'step3_filtered.png')
#
# # ---------- 主流程 ----------
# def main(image_path, save_dir):
#     if not os.path.exists(image_path):
#         print(f"Error: File {image_path} does not exist.")
#         return
#
#     os.makedirs(save_dir, exist_ok=True)
#     # Step 1: 去模糊
#     fuzzy_path = fuzzy_image_transform(image_path, save_dir)
#     print(f"Step 1 Done: {fuzzy_path}")
#
#     # Step 2: 高频增强
#     highpass_path = highpass_process(fuzzy_path, save_dir)
#     print(f"Step 2 Done: {highpass_path}")
#
#     # Step 3: 二值化+中值滤波+形态学开运算+连通域分析
#     filtered_path = bin_and_connect(highpass_path, save_dir)
#     print(f"Step 3 Done: {filtered_path}")
#
# # ---------- 使用方法 ----------
# if __name__ == "__main__":
#     # 修改为你的图片路径和保存目录
#     image_path = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\Letter_test_images\img_6.png"
#     save_dir = r"D:\PayCard_Detection\paycard_dectection_test1\Qumohu\process_results_method_1\test_debug_images_9"
#     main(image_path, save_dir)












############################这个代码是批量化处理图片，进行一个去模糊操作##################
#######################这个算法的核心思想是先用“模糊隶属度”把每个像素分解为暗/灰/亮三类
# #####################并通过重心解模糊实现平滑的分段线性对比拉伸，使灰度结构更清晰、分布更可分；
# #####################再在频域采用高频提升滤波（低频适度保留、高频放大）强化边缘与细节；
# ####################最后结合中值滤波、形态学开运算和连通域面积过滤，稳定地去除噪声与碎片、保留主要目标，从而形成一条“对比规范化 → 细节增强 → 稳健分割”的端到端预处理与提取流程

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_ubyte
from skimage.exposure import histogram

# ---------- Step 1: 去模糊（模糊知识变换） ----------
def fuzzy_knowledge(intensity):
    if intensity <= 0.27:
        drak = 1.0
    elif intensity >= 0.5:
        drak = 0.0
    else:
        drak = (0.5 - intensity) / 0.22

    if intensity >= 0.72:
        brig = 1.0
    elif intensity <= 0.5:
        brig = 0.0
    else:
        brig = (intensity - 0.5) / 0.22

    if intensity >= 0.72 or intensity <= 0.27:
        gray = 0.0
    elif intensity <= 0.5:
        gray = (intensity - 0.27) / 0.22
    else:
        gray = (0.72 - intensity) / 0.22

    return drak, gray, brig

def fuzzy_image_transform(image_path, save_dir):
    f = io.imread(image_path)
    f = img_as_float(f)
    if f.ndim == 3:
        f = f.mean(axis=2)
    M, N = f.shape
    g = np.zeros((M, N))

    for x in range(M):
        for y in range(N):
            drak, gray, brig = fuzzy_knowledge(f[x, y])
            g[x, y] = (drak * 0 + gray * 0.5 + brig * 1) / (drak + gray + brig)

    # 保存去模糊结果
    fuzzy_path = os.path.join(save_dir, 'step1_fuzzy.png')
    io.imsave(fuzzy_path, img_as_ubyte(g))

    # 保存原图和处理结果对比图
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(f, cmap='gray', vmin=0, vmax=1)
    plt.title('Original Image')
    plt.subplot(1,2,2)
    plt.imshow(g, cmap='gray', vmin=0, vmax=1)
    plt.title('Fuzzy Result')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'step1_compare.png'))
    plt.close()

    # 保存直方图
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    h, bins = histogram(f)
    h = h / (M*N)
    plt.bar(bins, h, width=1/255)
    plt.title('Histogram: Original')
    plt.subplot(1,2,2)
    h, bins = histogram(g)
    h = h / (M*N)
    plt.bar(bins, h, width=1/255)
    plt.title('Histogram: Fuzzy')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'step1_hist.png'))
    plt.close()

    return fuzzy_path

# ---------- Step 2: 高频增强 ----------
def manualShiftDFT(complexI):
    M, N = complexI.shape[:2]
    complexI_shifted = np.zeros_like(complexI)
    for i in range(M):
        for j in range(N):
            newRow = i + M//2 if i < M//2 else i - M//2
            newCol = j + N//2 if j < N//2 else j - N//2
            complexI_shifted[newRow, newCol] = complexI[i, j]
    return complexI_shifted

def lvboqi(M, N, D0=40):
    H = np.zeros((M, N), np.float64)
    for i in range(M):
        for j in range(N):
            D = np.sqrt((i - M/2)**2 + (j - N/2)**2)
            H[i, j] = np.exp(-(D**2) / (2 * (D0**2)))
    return H

def gaotonglvbo(g_shifted, one_minus_h):
    result = np.zeros_like(g_shifted)
    for i in range(g_shifted.shape[0]):
        for j in range(g_shifted.shape[1]):
            h_val = one_minus_h[i, j]
            result[i, j, 0] = g_shifted[i, j, 0] * h_val
            result[i, j, 1] = g_shifted[i, j, 1] * h_val
    return result

def gaotong(f):
    m, n = f.shape
    M1 = cv2.getOptimalDFTSize(m)
    N1 = cv2.getOptimalDFTSize(n)
    padded = cv2.copyMakeBorder(f, 0, M1-m, 0, N1-n, cv2.BORDER_CONSTANT, value=0)

    I = np.float64(padded)
    g = cv2.dft(I, flags=cv2.DFT_COMPLEX_OUTPUT)
    g_shifted = manualShiftDFT(g)
    M, N = g_shifted.shape[:2]
    H = lvboqi(M, N)
    F = 0.5 + 0.75 * (1 - H)
    result_highpass2 = gaotonglvbo(g_shifted, F)
    G_shifted = manualShiftDFT(result_highpass2)
    J2 = cv2.idft(G_shifted, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    J3 = J2[:m, :n]
    J4 = np.abs(J3)
    J4 = np.clip(J4, 0, 255).astype(np.uint8)
    return J4

def highpass_process(fuzzy_path, save_dir):
    img = cv2.imread(fuzzy_path, 0)
    result = gaotong(img)
    highpass_path = os.path.join(save_dir, 'step2_highpass.png')
    cv2.imwrite(highpass_path, result)

    # 保存对比图
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Fuzzy Input')
    plt.subplot(1,2,2)
    plt.imshow(result, cmap='gray')
    plt.title('Highpass Result')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'step2_compare.png'))
    plt.close()

    return highpass_path

# ---------- Step 3: 二值化+中值滤波+形态学开运算+连通域分析 ----------
def bin_and_connect(highpass_path, save_dir):
    img = cv2.imread(highpass_path, 0)
    img_blur = cv2.medianBlur(img, 3)  # 先中值滤波
    _, binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学开运算去毛刺
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)
    filtered = np.zeros_like(binary_clean)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 100:
            filtered[labels == i] = 255

    cv2.imwrite(os.path.join(save_dir, 'step3_binary.png'), binary)
    cv2.imwrite(os.path.join(save_dir, 'step3_binary_clean.png'), binary_clean)
    cv2.imwrite(os.path.join(save_dir, 'step3_filtered.png'), filtered)

    # 可视化对比
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(binary, cmap='gray'); plt.title('Binary')
    plt.subplot(1,3,2)
    plt.imshow(binary_clean, cmap='gray'); plt.title('Morph Open')
    plt.subplot(1,3,3)
    plt.imshow(filtered, cmap='gray'); plt.title('Filtered')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'step3_compare.png'))
    plt.close()

    return os.path.join(save_dir, 'step3_filtered.png')


def batch_main(image_dir, save_dir):
    if not os.path.exists(image_dir):
        print(f"Error: Folder {image_dir} does not exist.")
        return

    os.makedirs(save_dir, exist_ok=True)
    image_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_exts)]
    print(f"共找到 {len(image_files)} 张图片，开始批量处理...")

    for img_file in image_files:
        image_path = os.path.join(image_dir, img_file)
        img_name = os.path.splitext(img_file)[0]
        img_save_dir = os.path.join(save_dir, img_name)
        os.makedirs(img_save_dir, exist_ok=True)
        try:
            # Step 1: 去模糊
            fuzzy_path = fuzzy_image_transform(image_path, img_save_dir)
            print(f"[{img_file}] Step 1 Done: {fuzzy_path}")

            # Step 2: 高频增强
            highpass_path = highpass_process(fuzzy_path, img_save_dir)
            print(f"[{img_file}] Step 2 Done: {highpass_path}")

            # Step 3: 二值化+中值滤波+形态学开运算+连通域分析
            filtered_path = bin_and_connect(highpass_path, img_save_dir)
            print(f"[{img_file}] Step 3 Done: {filtered_path}")
        except Exception as e:
            print(f"[{img_file}] 处理失败: {e}")

    print("批量处理完成！")

if __name__ == "__main__":
    # 修改为你的图片文件夹和总保存目录
    image_dir = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\Letter_test_images"
    save_dir = r"D:\PayCard_Detection\paycard_dectection_test1\Qumohu\process_results_method_1\test_debug_images_batch_1"
    batch_main(image_dir, save_dir)


# # ---------- 主流程 ----------
# def main(image_path, save_dir):
#     if not os.path.exists(image_path):
#         print(f"Error: File {image_path} does not exist.")
#         return
#
#     os.makedirs(save_dir, exist_ok=True)
#     # Step 1: 去模糊
#     fuzzy_path = fuzzy_image_transform(image_path, save_dir)
#     print(f"Step 1 Done: {fuzzy_path}")
#
#     # Step 2: 高频增强
#     highpass_path = highpass_process(fuzzy_path, save_dir)
#     print(f"Step 2 Done: {highpass_path}")
#
#     # Step 3: 二值化+中值滤波+形态学开运算+连通域分析
#     filtered_path = bin_and_connect(highpass_path, save_dir)
#     print(f"Step 3 Done: {filtered_path}")
#
# # ---------- 使用方法 ----------
# if __name__ == "__main__":
#     # 修改为你的图片路径和保存目录
#     image_path = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\Letter_test_images\img_6.png"
#     save_dir = r"D:\PayCard_Detection\paycard_dectection_test1\Qumohu\process_results_method_1\test_debug_images_9"
#     main(image_path, save_dir)






