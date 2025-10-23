
###########################这个是对图片中的横条区域进行一个检测############################
import cv2
import numpy as np

def detect_middle_bar_projection(path, dilate_iter=1, gap_drop_ratio=0.5,
                                 return_debug=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    # Otsu 二值
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    bin_ = (th > 0).astype(np.uint8)

    # 若是细线边缘图，可膨胀让其“填充”
    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        for _ in range(dilate_iter):
            bin_ = cv2.dilate(bin_, kernel)

    h, w = bin_.shape
    row_gap = []
    for y in range(h):
        xs = np.where(bin_[y] > 0)[0]
        if xs.size == 0:
            row_gap.append((y, w))
            continue
        runs = []
        prev = xs[0]
        runs.append([prev, prev])
        for x in xs[1:]:
            if x == prev + 1:
                runs[-1][1] = x
            else:
                runs.append([x, x])
            prev = x
        gaps = []
        for i in range(len(runs)-1):
            gap = runs[i+1][0] - runs[i][1] - 1
            gaps.append(gap)
        max_gap = max(gaps) if gaps else 0
        row_gap.append((y, max_gap))

    gaps_arr = np.array([g for _, g in row_gap], dtype=float)

    # 平滑
    kernel_size = max(3, h//80)
    if kernel_size % 2 == 0:
        kernel_size += 1
    gaps_sm = cv2.GaussianBlur(gaps_arr.reshape(-1,1), (1, kernel_size), 0).ravel()

    baseline = np.percentile(gaps_sm, 90)
    threshold_gap = baseline * gap_drop_ratio

    bar_rows = np.where(gaps_sm < threshold_gap)[0]
    if bar_rows.size == 0:
        return None

    # 连续行分组
    groups = []
    cur = [bar_rows[0]]
    for yv in bar_rows[1:]:
        if yv == cur[-1] + 1:
            cur.append(yv)
        else:
            groups.append(cur)
            cur = [yv]
    groups.append(cur)

    main_group = max(groups, key=lambda g: len(g))
    y_top = main_group[0]
    y_bottom = main_group[-1]
    y_center = (y_top + y_bottom)/2
    h_norm = y_center / (h - 1)

    # 紧致左右边界
    xs_all = []
    for yv in main_group:
        xs = np.where(bin_[yv] > 0)[0]
        if xs.size:
            xs_all.append(xs)
    if xs_all:
        xs_concat = np.concatenate(xs_all)
        x_min = int(xs_concat.min())
        x_max = int(xs_concat.max())
    else:
        x_min, x_max = 0, w-1

    result = {
        "y_center": y_center,
        "y_range": (int(y_top), int(y_bottom)),
        "normalized_height": h_norm,
        "band_height": y_bottom - y_top + 1,
        "x_range": (x_min, x_max),
        "baseline_gap": baseline,
        "used_gap_threshold": threshold_gap,
        "image_shape": (h, w)
    }

    if return_debug:
        result["gaps_raw"] = gaps_arr
        result["gaps_smooth"] = gaps_sm

    return result, img, bin_


def visualize_middle_bar_box(path, save_path=None,
                             mode='tight', color=(255,0,0), thickness=2,
                             draw_center_line=True):
    """
    mode:
      'tight'  使用检测得到的 x_min~x_max
      'full'   使用整幅宽度 0~w-1
    """
    det = detect_middle_bar_projection(path, dilate_iter=1,
                                       gap_drop_ratio=0.5,
                                       return_debug=False)
    if det is None:
        print("未检测到中间横条")
        return None
    info, gray, _ = det
    (h, w) = info["image_shape"]
    y_top, y_bottom = info["y_range"]
    x_min, x_max = info["x_range"]

    if mode == 'full':
        x0, x1 = 0, w - 1
    else:
        x0, x1 = x_min, x_max

    y_center = int(round(info["y_center"]))

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 画蓝色矩形框 (BGR: 蓝色=(255,0,0))
    cv2.rectangle(vis, (x0, y_top), (x1, y_bottom), color, thickness)

    if draw_center_line:
        cv2.line(vis, (x0, y_center), (x1, y_center), (0,255,255), 1)

    text = f"y_center={y_center}, h_norm={info['normalized_height']:.3f}"
    cv2.putText(vis, text, (5, max(15, y_top-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, vis)
        print("已保存:", save_path)

    return vis, info


# 示例
if __name__ == "__main__":
    img_vis, info = visualize_middle_bar_box(
        r"D:\PayCard_Detection\paycard_dectection_test1\Muban\img_6.png",
        save_path=r"D:\PayCard_Detection\paycard_dectection_test1\Muban\img_6_roi_box.png",
        mode='tight',   # 或 'full'
        color=(255,0,0),
        thickness=1
    )
    print(info)

