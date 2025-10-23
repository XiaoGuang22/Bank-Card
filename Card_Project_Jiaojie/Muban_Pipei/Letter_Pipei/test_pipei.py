###########################记录当个字母字符的匹配时间########################
###########################这个是做了一个H和N和M的二次匹配验证代码，后期可能这方面需要改进################
#######################目前主要匹配的代码是运行这个#############################
########################模板图片文件有两个：Letter_muban_images(这个是53个模板，每个字母两个，更新之后的）#############
########################Letter_muban_images_test_all（这个里面有大概78个模板，每个字符多个，之前没有更新之后的）###########


import cv2
import numpy as np
import os
import uuid
import time

# ===================== 横条检测相关函数 =====================
def detect_middle_bar_projection(path, dilate_iter=1, gap_drop_ratio=0.5, return_debug=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    bin_ = (th > 0).astype(np.uint8)
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
    kernel_size = max(3, h//80)
    if kernel_size % 2 == 0:
        kernel_size += 1
    gaps_sm = cv2.GaussianBlur(gaps_arr.reshape(-1,1), (1, kernel_size), 0).ravel()
    baseline = np.percentile(gaps_sm, 90)
    threshold_gap = baseline * gap_drop_ratio
    bar_rows = np.where(gaps_sm < threshold_gap)[0]
    if bar_rows.size == 0:
        return None
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

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_img(path, img):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def crop_to_content(img, threshold=10):
    rows = np.any(img > threshold, axis=1)
    cols = np.any(img > threshold, axis=0)
    if not np.any(rows) or not np.any(cols):
        return img
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return img[y_min:y_max + 1, x_min:x_max + 1]

def center_resize(img, target_shape, bg=0):
    h, w = img.shape
    th, tw = target_shape
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.ones(target_shape, dtype=np.uint8) * bg
    y0 = (th - new_h) // 2
    x0 = (tw - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas

def find_largest_contour(img):
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def load_templates(template_dir):
    templates = {}
    for fname in os.listdir(template_dir):
        if not fname.lower().endswith('.png'):
            continue
        label = os.path.splitext(fname)[0][0].upper()
        img = cv2.imread(os.path.join(template_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None and label.isalpha():
            if label not in templates:
                templates[label] = []
            templates[label].append(img)
    return templates

def segment_characters(bw_img, min_area=30):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_img, connectivity=8)
    candidates = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        roi = bw_img[y:y + h, x:x + w]
        candidates.append({
            "bbox": (x, y, w, h),
            "roi": roi,
            "id": uuid.uuid4().hex[:8]
        })
    candidates.sort(key=lambda c: c["bbox"][0])
    return candidates

def ncc_score(a, b):
    A = a.astype(np.float32)
    B = b.astype(np.float32)
    meanA, meanB = A.mean(), B.mean()
    num = np.sum((A - meanA) * (B - meanB))
    den = np.sqrt(np.sum((A - meanA) ** 2) * np.sum((B - meanB) ** 2))
    if den == 0:
        return -1.0
    return num / den

# ===================== 记录匹配用时 =====================
def match_char_to_templates(char_roi, templates, debug_dir, char_idx):
    start_time = time.time()  # 开始计时
    best_label = None
    best_score = -1.0
    best_tpl_idx = 0
    scores = []
    for label, tpl_imgs in templates.items():
        for tpl_idx, tpl_img in enumerate(tpl_imgs):
            roi_crop = crop_to_content(char_roi)
            char_resized = center_resize(roi_crop, tpl_img.shape, bg=0)
            score = ncc_score(char_resized, tpl_img)
            scores.append((label, score, tpl_idx))
            save_img(os.path.join(debug_dir, f"char{char_idx:02d}_{label}_{tpl_idx}_resized.png"), char_resized)
            if score > best_score:
                best_score = score
                best_label = label
                best_tpl_idx = tpl_idx
    scores.sort(key=lambda x: x[1], reverse=True)
    elapsed_ms = int((time.time() - start_time) * 1000)  # 匹配耗时（毫秒）
    return best_label, best_score, scores, best_tpl_idx, elapsed_ms

def recognize_image(image_path, template_dir, debug_dir="debug_outputs", min_area=30):
    ensure_dir(debug_dir)
    raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    save_img(os.path.join(debug_dir, "01_raw.png"), raw)
    _, bw = cv2.threshold(raw, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # _, bw = cv2.threshold(raw, 190, 255, cv2.THRESH_BINARY)
    save_img(os.path.join(debug_dir, "02_bw.png"), bw)
    # chars = segment_characters(bw_eroded, min_area=min_area)
    chars = segment_characters(bw, min_area=min_area)
    vis_boxes = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for idx, c in enumerate(chars):
        x, y, w, h = c["bbox"]
        cv2.rectangle(vis_boxes, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(vis_boxes, str(idx), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    save_img(os.path.join(debug_dir, "03_seg_boxes.png"), vis_boxes)
    templates = load_templates(template_dir)
    results = []
    roi_dir = os.path.join(debug_dir, "chars")
    ensure_dir(roi_dir)
    for idx, c in enumerate(chars):
        roi_path = os.path.join(roi_dir, f"char{idx:02d}_roi.png")
        save_img(roi_path, c["roi"])
        best_label, best_score, scores, best_tpl_idx, match_time_ms = match_char_to_templates(c["roi"], templates, roi_dir, idx)
        contour_scores = {}
        confirmed_label = None
        contour_score = None
        if best_label in ["H", "N", "M"]:
            normalized_img = center_resize(crop_to_content(c["roi"]), templates[best_label][best_tpl_idx].shape, bg=0)
            min_contour_score = 1e9
            min_contour_label = None
            min_contour_tpl_idx = -1
            for label in ["H", "N", "M"]:
                if label in templates:
                    for tpl_idx, tpl_img in enumerate(templates[label]):
                        tpl_bin = (tpl_img > 128).astype(np.uint8) * 255
                        tpl_contour = find_largest_contour(tpl_bin)
                        char_bin = (normalized_img > 128).astype(np.uint8) * 255
                        char_contour = find_largest_contour(char_bin)
                        if tpl_contour is not None and char_contour is not None:
                            score = cv2.matchShapes(char_contour, tpl_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                            if label not in contour_scores:
                                contour_scores[label] = []
                            contour_scores[label].append(score)
                            if score < min_contour_score:
                                min_contour_score = score
                                min_contour_label = label
                                min_contour_tpl_idx = tpl_idx
                        else:
                            if label not in contour_scores:
                                contour_scores[label] = []
                            contour_scores[label].append(None)
            if min_contour_label is not None:
                confirmed_label = min_contour_label
                contour_score = min_contour_score
                best_label = confirmed_label
        else:
            contour_scores = {}
        if best_label in ["H", "M"]:
            roi_temp_path = os.path.join(roi_dir, f"char{idx:02d}_roi_for_bar.png")
            save_img(roi_temp_path, crop_to_content(c["roi"]))
            try:
                bar_result = detect_middle_bar_projection(roi_temp_path, dilate_iter=1, gap_drop_ratio=0.5)
            except Exception as e:
                bar_result = None
            BAR_RATIO_THRESHOLD = 0.3
            if bar_result is None or bar_result[0] is None:
                pass
            else:
                bar_info = bar_result[0]
                band_height = bar_info["band_height"]
                h_total = bar_info["image_shape"][0]
                band_ratio = band_height / h_total if h_total > 0 else 0
                if band_ratio < BAR_RATIO_THRESHOLD:
                    best_label = "H"
                else:
                    best_label = "M"
                roi_img = crop_to_content(c["roi"])
                x_min, x_max = bar_info["x_range"]
                y_top, y_bottom = bar_info["y_range"]
                h_roi, w_roi = roi_img.shape[:2]
                x_min = max(0, min(x_min, w_roi - 1))
                x_max = max(0, min(x_max, w_roi - 1))
                y_top = max(0, min(y_top, h_roi - 1))
                y_bottom = max(0, min(y_bottom, h_roi - 1))
                bar_crop = roi_img[y_top:y_bottom + 1, x_min:x_max + 1]
                bar_crop_path = os.path.join(roi_dir, f"char{idx:02d}_bar_region.png")
                save_img(bar_crop_path, bar_crop)
        results.append({
            "bbox": c["bbox"],
            "label": best_label,
            "score": best_score,
            "topk": scores[:5],
            "contour_scores": contour_scores,
            "match_elapsed_ms": match_time_ms  # 新增时间字段
        })
    annotated = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    for idx, r in enumerate(results):
        x, y, w, h = r["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(annotated, f"{r['label']}:{r['score']:.2f}", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    save_img(os.path.join(debug_dir, "04_annotated.png"), annotated)
    recognized_str = "".join(r["label"] if r["label"] else "?" for r in results)
    with open(os.path.join(debug_dir, "final_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"recognized={recognized_str}\n")
        for idx, r in enumerate(results):
            f.write(f"{idx}\t{r['bbox']}\t{r['label']}\t{r['score']:.4f}\t{r['match_elapsed_ms']}ms\n")
            if r["contour_scores"]:
                for label in ["H", "M", "N"]:
                    scores = r["contour_scores"].get(label, [])
                    for i, score in enumerate(scores):
                        if score is not None:
                            f.write(f"\tcontour_{label}_{i}={score:.4f}\n")
                        else:
                            f.write(f"\tcontour_{label}_{i}=None\n")
    return recognized_str, results

# ===================== 批量识别+计时 =====================
def batch_recognize_images(image_dir, template_dir, batch_debug_dir="batch_debug_outputs", min_area=30):
    ensure_dir(batch_debug_dir)
    image_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_exts)]
    all_results_path = os.path.join(batch_debug_dir, "all_results.txt")
    with open(all_results_path, "w", encoding="utf-8") as all_f:
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            debug_dir = os.path.join(batch_debug_dir, os.path.splitext(img_file)[0])
            start_time = time.time()
            recognized_str, results = recognize_image(img_path, template_dir, debug_dir, min_area=min_area)
            end_time = time.time()
            elapsed_ms = int((end_time - start_time) * 1000)
            all_f.write(f"{img_file}\t{recognized_str}\t{elapsed_ms}ms\n")
            for idx, r in enumerate(results):
                all_f.write(f"{img_file}\t{idx}\t{r['bbox']}\t{r['label']}\t{r['score']:.4f}\t{r['match_elapsed_ms']}ms\n")
            print(f"{img_file} 识别完成，用时：{elapsed_ms} ms")
    print("批量识别完成，总结果保存在：", all_results_path)

# ===================== 主调用入口 =====================
if __name__ == "__main__":
    TEMPLATE_DIR = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\Letter\Muban_images"
    IMAGE_DIR = r"D:\PayCard_Detection\Card_Project_Jiaojie\Muban_Pipei\Letter_Pipei\example_images"
    BATCH_DEBUG_DIR = "batch_debug_outputs_2"
    batch_recognize_images(IMAGE_DIR, TEMPLATE_DIR, BATCH_DEBUG_DIR)
