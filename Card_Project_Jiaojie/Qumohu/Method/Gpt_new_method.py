import os, cv2, numpy as np, matplotlib.pyplot as plt

input_path  = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\Letter\test_images\img_22.png"  # 输入图像路径
output_dir  = 'full_enhance_outputs_1'

force_invert = None
bg_gauss_sigma = 14
diff_large_sigma = 10.0
diff_small_sigma = 1.3
halo_diff_clip = 0.0  # 一般保持0

dehalo_merge_weight_bg = 0.55
dehalo_merge_weight_diff = 0.45

stretch_low_pct  = 1.0
stretch_high_pct = 99.2

use_CLAHE   = True
clahe_clip  = 2.2
clahe_tile  = (8,8)

erode_iterations = 2
alpha_gamma      = 1.6
alpha_min_cut    = 0.03

sharpen_scales    = [1.0, 2.0, 4.0]
sharpen_amounts   = [0.9, 0.5, 0.3]
final_sharpen_mix = 0.65
edge_weight_power = 0.55
dehalo_clip       = 0.015

premultiply_alpha = False

save_extra_debug  = True


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def auto_invert_needed(gray_u8):
    mean_all = gray_u8.mean()
    hi = np.percentile(gray_u8, 95)
    ratio_hi = (gray_u8 >= hi).mean()
    return (mean_all > 150 and ratio_hi > 0.25)


def stretch_percentile(img_01, low_pct, high_pct):
    lo, hi = np.percentile(img_01, (low_pct, high_pct))
    out = (img_01 - lo) / (hi - lo + 1e-6)
    return np.clip(out, 0, 1)


def apply_CLAHE(gray_01):
    g8 = (gray_01 * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    out = clahe.apply(g8)
    return out.astype(np.float32) / 255.0


def build_soft_alpha(gray_01):
    g8 = (gray_01 * 255).astype(np.uint8)
    _, outer = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.erode(outer, kernel, iterations=erode_iterations)
    if core.sum() == 0:
        core = outer.copy()
    core_bin = (core > 0).astype(np.uint8)
    outer_bin = (outer > 0).astype(np.uint8)
    ring = outer_bin - core_bin
    ring[ring < 0] = 0
    inv_core = 1 - core_bin
    dist = cv2.distanceTransform(inv_core, cv2.DIST_L2, 3)
    dist_ring = dist * ring
    maxd = dist_ring.max() if dist_ring.max() > 0 else 1.0
    alpha = np.zeros_like(gray_01)
    alpha[core_bin == 1] = 1.0
    rp = ring == 1
    alpha[rp] = 1.0 - dist_ring[rp] / (maxd + 1e-6)
    alpha = np.clip(alpha, 0, 1)
    if alpha_gamma != 1.0:
        alpha = alpha ** alpha_gamma
    alpha[alpha < alpha_min_cut] = 0
    return alpha, core_bin, ring, dist_ring / (maxd + 1e-6)


def multi_scale_sharpen(gray_01, scales, amounts):
    accum = np.zeros_like(gray_01)
    for s, w in zip(scales, amounts):
        blur = cv2.GaussianBlur(gray_01, (0, 0), s)
    detail = gray_01 - blur
    accum += w * detail
    sharpened = np.clip(gray_01 + accum, 0, 1)
    return sharpened, accum


def edge_weight_map(gray_01):
    gx = cv2.Sobel(gray_01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_01, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx + gy)
    mag /= (mag.max() + 1e-6)
    mag = cv2.GaussianBlur(mag, (3, 3), 0)
    w = np.clip(mag, 0, 1) ** edge_weight_power
    return w


def prevent_halo(orig, sharpened):
    diff = sharpened - orig
    over_pos = diff > dehalo_clip
    over_neg = diff < -dehalo_clip
    sharpened[over_pos] = orig[over_pos] + dehalo_clip
    sharpened[over_neg] = orig[over_neg] - dehalo_clip
    return np.clip(sharpened, 0, 1)


def save_gray01(path, g):
    cv2.imwrite(path, (np.clip(g, 0, 1) * 255).astype(np.uint8))


def main():
    ensure_dir(output_dir)
    img = read_image(input_path)
    H,W = img.shape[:2]
    cv2.imwrite(os.path.join(output_dir,'01_input.png'), img)
    # Lab 亮度
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0
    save_gray01(os.path.join(output_dir, '10_L_original.png'), L)

    # 自动或强制反色
    if force_invert is None:
        need_inv = auto_invert_needed((L * 255).astype(np.uint8))
    else:
        need_inv = bool(force_invert)
    L_proc = 1.0 - L if need_inv else L.copy()
    save_gray01(os.path.join(output_dir, '11_L_used.png'), L_proc)

    # 轻去噪
    L_denoise = cv2.GaussianBlur(L_proc, (0, 0), 0.6)
    save_gray01(os.path.join(output_dir, '12_L_denoise.png'), L_denoise)

    # 方法A：背景减法去光晕
    bg = cv2.GaussianBlur(L_denoise, (0, 0), bg_gauss_sigma)
    L_dehalo_bg = np.clip(L_denoise - (bg - bg.min()), 0, 1)
    L_dehalo_bg = stretch_percentile(L_dehalo_bg, stretch_low_pct, stretch_high_pct)
    save_gray01(os.path.join(output_dir, '20_L_dehalo_bg.png'), L_dehalo_bg)
    save_gray01(os.path.join(output_dir, '21_bg_large_blur.png'), bg / (bg.max() + 1e-6))

    # 方法B：差分去光晕
    blur_large = cv2.GaussianBlur(L_denoise, (0, 0), diff_large_sigma)
    blur_small = cv2.GaussianBlur(L_denoise, (0, 0), diff_small_sigma)
    diff_halo = np.clip(blur_large - blur_small - halo_diff_clip, 0, 1)
    L_dehalo_diff = np.clip(L_denoise - diff_halo, 0, 1)
    L_dehalo_diff = stretch_percentile(L_dehalo_diff, stretch_low_pct, stretch_high_pct)
    save_gray01(os.path.join(output_dir, '21_L_dehalo_diff.png'), L_dehalo_diff)
    save_gray01(os.path.join(output_dir, '22_diff_halo_est.png'), diff_halo)

    # 融合
    wa = max(dehalo_merge_weight_bg, 0.0)
    wb = max(dehalo_merge_weight_diff, 0.0)
    if wa + wb == 0: wa, wb = 1, 1
    L_dehalo = (wa * L_dehalo_bg + wb * L_dehalo_diff) / (wa + wb)
    save_gray01(os.path.join(output_dir, '30_L_after_dehalo.png'), L_dehalo)

    # 再拉伸
    L_dehalo_stretch = stretch_percentile(L_dehalo, stretch_low_pct, stretch_high_pct)
    save_gray01(os.path.join(output_dir, '31_L_dehalo_stretch.png'), L_dehalo_stretch)

    # 对比增强
    L_contrast = apply_CLAHE(L_dehalo_stretch) if use_CLAHE else L_dehalo_stretch
    save_gray01(os.path.join(output_dir, '40_L_contrast.png'), L_contrast)

    # Soft Alpha
    alpha_soft, core_mask, ring_mask, dist_norm = build_soft_alpha(L_contrast)
    save_gray01(os.path.join(output_dir, '50_alpha_soft.png'), alpha_soft)
    cv2.imwrite(os.path.join(output_dir, '51_mask_core.png'), core_mask * 255)
    cv2.imwrite(os.path.join(output_dir, '52_mask_ring.png'), ring_mask * 255)
    save_gray01(os.path.join(output_dir, '53_dist_norm.png'), dist_norm)

    # 边缘权重 + 多尺度锐化
    edge_w = edge_weight_map(L_contrast)
    save_gray01(os.path.join(output_dir, '80_edge_weight.png'), edge_w)

    sharp_raw, sharp_layer = multi_scale_sharpen(L_contrast, sharpen_scales, sharpen_amounts)
    sharp_weighted = np.clip(L_contrast * (1 - edge_w) + sharp_raw * edge_w, 0, 1)
    L_sharp = L_contrast * (1 - final_sharpen_mix) + sharp_weighted * final_sharpen_mix
    L_sharp = prevent_halo(L_contrast, L_sharp)
    save_gray01(os.path.join(output_dir, '60_L_sharp_final.png'), L_sharp)

    layer_norm = sharp_layer - sharp_layer.min()
    layer_norm /= (layer_norm.max() + 1e-6)
    save_gray01(os.path.join(output_dir, '81_sharpen_layer.png'), layer_norm)

    # 反色恢复
    L_out = 1.0 - L_sharp if need_inv else L_sharp

    # 彩色恢复
    lab_out = lab.copy()
    lab_out[:, :, 0] = (np.clip(L_out, 0, 1) * 255).astype(np.uint8)
    color_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

    # 保存灰度 / 彩色
    save_gray01(os.path.join(output_dir, '70_enhanced_gray.png'), L_out)
    cv2.imwrite(os.path.join(output_dir, '71_enhanced_color.png'),
                (np.clip(color_out, 0, 1) * 255).astype(np.uint8))

    # 透明 RGBA
    alpha_u8 = (alpha_soft * 255).astype(np.uint8)
    rgb_u8 = (np.clip(color_out, 0, 1) * 255).astype(np.uint8)
    if premultiply_alpha:
        rgb_premul = (rgb_u8.astype(np.float32) * (alpha_soft[..., None])).astype(np.uint8)
        rgba = cv2.cvtColor(rgb_premul, cv2.COLOR_BGR2BGRA)
    else:
        rgba = cv2.cvtColor(rgb_u8, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha_u8
    cv2.imwrite(os.path.join(output_dir, '72_enhanced_rgba.png'), rgba)
    cv2.imwrite(os.path.join(output_dir, '73_alpha.png'), alpha_u8)

    # 棋盘预览
    if save_extra_debug:
        tile = 10
        yy, xx = np.indices((H, W))
        board = ((xx // tile + yy // tile) % 2).astype(np.uint8)
        board_rgb = (board * 255).astype(np.uint8)
        board_rgb[board_rgb == 255] = 220
        board_rgb[board_rgb == 0] = 120
        board_rgb = cv2.cvtColor(board_rgb, cv2.COLOR_GRAY2BGR)
        a = alpha_soft[..., None]
        comp = (rgb_u8.astype(np.float32) * a + board_rgb * (1 - a)).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, '74_checkboard_preview.png'), comp)

    # 流程概览
    steps = [
        ('原亮度', L),
        ('使用亮度(可能反色)', L_proc),
        ('去噪', L_denoise),
        ('背景去光影', L_dehalo_bg),
        ('差分去光影', L_dehalo_diff),
        ('融合去光影', L_dehalo),
        ('拉伸', L_dehalo_stretch),
        ('对比增强', L_contrast),
        ('Soft Alpha', alpha_soft),
        ('边缘权重', edge_w),
        ('锐化后', L_sharp),
        ('最终输出亮度', L_out)
    ]
    cols = 4
    rows = int(np.ceil(len(steps) / cols))
    plt.figure(figsize=(14, 3.2 * rows))
    for i, (title, im) in enumerate(steps, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(im, cmap='gray', vmin=0, vmax=1)
        plt.title(title, fontsize=10)
        plt.axis('off')
    plt.suptitle(f'文本增强流程 (invert={need_inv})', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'pipeline_overview.png'), dpi=120)
    plt.close()

    print('[DONE] 全部完成, 输出目录:', output_dir)
    print('主要结果: 72_enhanced_rgba.png / 71_enhanced_color.png / pipeline_overview.png')

if __name__ == "__main__":
    main()
