import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_ubyte
from skimage.exposure import histogram
from skimage.filters import threshold_multiotsu
import cv2
import os

# ---------- Step 1: 自适应去模糊 ----------
def get_thresholds(f, method='percentile', p_low=10, p_high=90):
    f_valid = f[np.isfinite(f)]
    if f_valid.size == 0:
        return 0.27, 0.5, 0.72
    if method == 'percentile':
        l = float(np.percentile(f_valid, p_low))
        m = float(np.percentile(f_valid, 50))
        h = float(np.percentile(f_valid, p_high))
    elif method == 'multiotsu':
        t1, t2 = threshold_multiotsu(f_valid, classes=3)
        l, h = float(t1), float(t2)
        m = (l + h) / 2.0
    else:
        raise ValueError("method must be 'percentile' or 'multiotsu'")
    eps = 1e-6
    l = float(np.clip(l, 0.0, 1.0))
    h = float(np.clip(h, 0.0, 1.0))
    m = float(np.clip(m, l + eps, h - eps))
    if not (l < m < h):
        l, m, h = 0.27, 0.5, 0.72
    return l, m, h

def fuzzy_membership(img, l, m, h):
    d = np.where(img <= l, 1.0,
        np.where(img >= m, 0.0, (m - img) / (m - l + 1e-12)))
    b = np.where(img >= h, 1.0,
        np.where(img <= m, 0.0, (img - m) / (h - m + 1e-12)))
    g = np.where((img <= l) | (img >= h), 0.0,
        np.where(img <= m, (img - l) / (m - l + 1e-12),
                           (h - img) / (h - m + 1e-12)))
    return d, g, b

def fuzzy_image_transform(image_path, save_dir, method='multiotsu'):
    f = io.imread(image_path)
    f = img_as_float(f)
    if f.ndim == 3:
        f = f.mean(axis=2)
    M, N = f.shape
    l, m, h = get_thresholds(f, method=method)
    print(f"自适应阈值: l={l:.4f}, m={m:.4f}, h={h:.4f} (method={method})")
    drak, gray, brig = fuzzy_membership(f, l, m, h)
    weights_sum = drak + gray + brig + 1e-12
    g = (drak * 0.0 + gray * 0.5 + brig * 1.0) / weights_sum

    fuzzy_img_path = os.path.join(save_dir, 'step1_fuzzy.png')
    io.imsave(fuzzy_img_path, img_as_ubyte(g))

    # 保存对比图
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(f, cmap='gray', vmin=0, vmax=1)
    plt.title('Original')
    plt.subplot(1,2,2)
    plt.imshow(g, cmap='gray', vmin=0, vmax=1)
    plt.title('Fuzzy Result')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'step1_compare.png'))
    plt.close()
    return fuzzy_img_path

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

def highpass_process(fuzzy_img_path, save_dir):
    img = cv2.imread(fuzzy_img_path, 0)
    result = gaotong(img)
    highpass_img_path = os.path.join(save_dir, 'step2_highpass.png')
    cv2.imwrite(highpass_img_path, result)

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
    return highpass_img_path

# ---------- Step 3: 二值化 + 连通域分析 ----------
def bin_and_connect(highpass_img_path, save_dir):
    img = cv2.imread(highpass_img_path, 0)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img_path = os.path.join(save_dir, 'step3_binary.png')
    cv2.imwrite(bin_img_path, binary)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 100:
            filtered[labels == i] = 255
    filtered_img_path = os.path.join(save_dir, 'step3_filtered.png')
    cv2.imwrite(filtered_img_path, filtered)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary')
    plt.subplot(1,2,2)
    plt.imshow(filtered, cmap='gray')
    plt.title('Filtered')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'step3_compare.png'))
    plt.close()
    return filtered_img_path

# ---------- 主流程 ----------
def main(image_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print("Step 1: 去模糊")
    fuzzy_img_path = fuzzy_image_transform(image_path, save_dir, method='multiotsu')
    print("Step 2: 高频增强")
    highpass_img_path = highpass_process(fuzzy_img_path, save_dir)
    print("Step 3: 二值化 + 连通域分析")
    filtered_img_path = bin_and_connect(highpass_img_path, save_dir)
    print("全部完成，结果已保存在：", save_dir)

# ---------- 用法 ----------
if __name__ == "__main__":
    image_path = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\Letter\test_images\img_31.png"
    save_dir = r"D:\PayCard_Detection\paycard_dectection_test1\Qumohu\process_results_method_2\test_debug_images_2"
    main(image_path, save_dir)
