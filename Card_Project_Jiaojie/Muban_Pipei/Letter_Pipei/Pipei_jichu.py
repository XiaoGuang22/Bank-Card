# ################这个可以实现基本的一个模板匹配识别，不过H和M和N很难区分出来###################
# import cv2
# import numpy as np
# import os
# import uuid
#
# def ensure_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#
# def save_img(path, img):
#     ensure_dir(os.path.dirname(path))
#     cv2.imwrite(path, img)
#
# # ---- 1. 加载模板库 ----
# def load_templates(template_dir):
#     templates = {}
#     for fname in os.listdir(template_dir):
#         if not fname.lower().endswith('.png'):
#             continue
#         label = os.path.splitext(fname)[0].upper()
#         img = cv2.imread(os.path.join(template_dir, fname), cv2.IMREAD_GRAYSCALE)
#         if img is not None and len(label) == 1 and label.isalpha():
#             templates[label] = img
#     return templates
#
# # ---- 2. 字符分割（连通域法） ----
# def segment_characters(bw_img, min_area=30):
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_img, connectivity=8)
#     candidates = []
#     for i in range(1, num_labels):
#         x, y, w, h, area = stats[i]
#         if area < min_area:
#             continue
#         roi = bw_img[y:y+h, x:x+w]
#         candidates.append({
#             "bbox": (x, y, w, h),
#             "roi": roi,
#             "id": uuid.uuid4().hex[:8]
#         })
#     candidates.sort(key=lambda c: c["bbox"][0])
#     return candidates
#
# # ---- 3. 匹配函数（resize到模板大小） ----
# def ncc_score(a, b):
#     A = a.astype(np.float32)
#     B = b.astype(np.float32)
#     meanA, meanB = A.mean(), B.mean()
#     num = np.sum((A - meanA) * (B - meanB))
#     den = np.sqrt(np.sum((A - meanA) ** 2) * np.sum((B - meanB) ** 2))
#     if den == 0:
#         return -1.0
#     return num / den
#
# def match_char_to_templates(char_roi, templates, debug_dir, char_idx):
#     best_label = None
#     best_score = -1.0
#     scores = []
#     for label, tpl_img in templates.items():
#         # resize char_roi 到模板大小
#         char_resized = cv2.resize(char_roi, tpl_img.shape[::-1], interpolation=cv2.INTER_LINEAR)
#         score = ncc_score(char_resized, tpl_img)
#         scores.append((label, score))
#         # 保存归一化后的图片
#         save_img(os.path.join(debug_dir, f"char{char_idx:02d}_{label}_resized.png"), char_resized)
#         if score > best_score:
#             best_score = score
#             best_label = label
#     scores.sort(key=lambda x: x[1], reverse=True)
#     return best_label, best_score, scores
#
# # ---- 4. 主识别流程 ----
# def recognize_image(image_path, template_dir, debug_dir="debug_outputs", min_area=30):
#     ensure_dir(debug_dir)
#     # step1: 读入原图
#     raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     save_img(os.path.join(debug_dir, "01_raw.png"), raw)
#     # step2: 二值化
#     _, bw = cv2.threshold(raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     save_img(os.path.join(debug_dir, "02_bw.png"), bw)
#     # step3: 分割字符区域
#     chars = segment_characters(bw, min_area=min_area)
#     # step4: 可视化分割结果
#     vis_boxes = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
#     for idx, c in enumerate(chars):
#         x, y, w, h = c["bbox"]
#         cv2.rectangle(vis_boxes, (x, y), (x+w, y+h), (0,0,255), 1)
#         cv2.putText(vis_boxes, str(idx), (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
#     save_img(os.path.join(debug_dir, "03_seg_boxes.png"), vis_boxes)
#     # step5: 加载模板
#     templates = load_templates(template_dir)
#     # step6: 循环匹配
#     results = []
#     roi_dir = os.path.join(debug_dir, "chars")
#     ensure_dir(roi_dir)
#     for idx, c in enumerate(chars):
#         roi_path = os.path.join(roi_dir, f"char{idx:02d}_roi.png")
#         save_img(roi_path, c["roi"])
#         best_label, best_score, scores = match_char_to_templates(c["roi"], templates, roi_dir, idx)
#         results.append({
#             "bbox": c["bbox"],
#             "label": best_label,
#             "score": best_score,
#             "topk": scores[:5]
#         })
#     # step7: 结果可视化
#     annotated = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
#     for idx, r in enumerate(results):
#         x, y, w, h = r["bbox"]
#         cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,0,255), 2)
#         cv2.putText(annotated, f"{r['label']}:{r['score']:.2f}", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
#     save_img(os.path.join(debug_dir, "04_annotated.png"), annotated)
#     # step8: 结果文本
#     recognized_str = "".join(r["label"] if r["label"] else "?" for r in results)
#     with open(os.path.join(debug_dir, "final_result.txt"), "w", encoding="utf-8") as f:
#         f.write(f"recognized={recognized_str}\n")
#         for idx, r in enumerate(results):
#             f.write(f"{idx}\t{r['bbox']}\t{r['label']}\t{r['score']:.4f}\n")
#     print("识别结果：", recognized_str)
#     return recognized_str, results
#
# # ---- 用法 ----
# if __name__ == "__main__":
#     TEMPLATE_DIR = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\Muban_images"
#     IMAGE_PATH = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\img_1.png"
#     DEBUG_DIR = "debug_outputs_test2"
#     recognize_image(IMAGE_PATH, TEMPLATE_DIR, DEBUG_DIR)
#
#







#######################这个代码通过计算H和M之间得横条得高度占整个字符高度得比例（H和M这两个字符进行多一个特殊得判别）
#######################如果这个比例小于0.3,那么判度为H，如果为M判度为M##################
#######################这个处理问题还是有点大，后期可以参考这个方法#################################
import cv2
import numpy as np
import os
import uuid

HM_BAR_RATIO_THRESHOLD = 0.3       # 判定阈值: band_height / char_height
HM_SCORE_CLOSE_DELTA = 0.1        # H 与 M 分数接近的阈值
HM_DILATE_ITER = 1
HM_GAP_DROP_RATIO = 0.5

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_img(path, img):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

# ---- 1. 加载模板库 ----
def load_templates(template_dir):
    templates = {}
    for fname in os.listdir(template_dir):
        if not fname.lower().endswith('.png'):
            continue
        # 只取文件名第一个字母作为label
        label = os.path.splitext(fname)[0][0].upper()
        img = cv2.imread(os.path.join(template_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None and label.isalpha():
            # 只保留第一个遇到的模板
            if label not in templates:
                templates[label] = img
    return templates



# ---- 2. 字符分割（连通域法） ----
def segment_characters(bw_img, min_area=30):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_img, connectivity=8)
    candidates = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        roi = bw_img[y:y+h, x:x+w]
        candidates.append({
            "bbox": (x, y, w, h),
            "roi": roi,
            "id": uuid.uuid4().hex[:8]
        })
    candidates.sort(key=lambda c: c["bbox"][0])
    return candidates

# ---- 3. 匹配函数（resize到模板大小） ----
def ncc_score(a, b):
    A = a.astype(np.float32)
    B = b.astype(np.float32)
    meanA, meanB = A.mean(), B.mean()
    num = np.sum((A - meanA) * (B - meanB))
    den = np.sqrt(np.sum((A - meanA) ** 2) * np.sum((B - meanB) ** 2))
    if den == 0:
        return -1.0
    return num / den

# def match_char_to_templates(char_roi, templates, debug_dir, char_idx):
#     best_label = None
#     best_score = -1.0
#     scores = []
#     for label, tpl_img in templates.items():
#         char_resized = cv2.resize(char_roi, tpl_img.shape[::-1], interpolation=cv2.INTER_LINEAR)
#         score = ncc_score(char_resized, tpl_img)
#         scores.append((label, score))
#         save_img(os.path.join(debug_dir, f"char{char_idx:02d}_{label}_resized.png"), char_resized)
#         if score > best_score:
#             best_score = score
#             best_label = label
#     scores.sort(key=lambda x: x[1], reverse=True)
#     return best_label, best_score, scores

# ##########进行腐蚀操作########################
def match_char_to_templates(char_roi, templates, debug_dir, char_idx,
                            roi_erode=False,
                            roi_erode_kernel=(3,3),
                            roi_erode_iter=1):
    # 可选 ROI 腐蚀
    if roi_erode:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, roi_erode_kernel)
        eroded = cv2.erode(char_roi, k, iterations=roi_erode_iter)
        save_img(os.path.join(debug_dir, f"char{char_idx:02d}_roi_eroded.png"), eroded)
        work_roi = eroded
    else:
        work_roi = char_roi

    best_label = None
    best_score = -1.0
    scores = []
    for label, tpl_img in templates.items():
        char_resized = cv2.resize(work_roi, tpl_img.shape[::-1], interpolation=cv2.INTER_LINEAR)
        score = ncc_score(char_resized, tpl_img)
        scores.append((label, score))
        save_img(os.path.join(debug_dir, f"char{char_idx:02d}_{label}_resized.png"), char_resized)
        if score > best_score:
            best_score = score
            best_label = label
    scores.sort(key=lambda x: x[1], reverse=True)
    return best_label, best_score, scores


# ---- 4.a 从数组检测中间横条（改造版） ----
def detect_middle_bar_projection_from_array(gray_img,
                                            dilate_iter=1,
                                            gap_drop_ratio=0.5):
    """
    输入: 灰度或二值字符 ROI (ndarray)
    返回:
      (result_dict, bin_img) 或 None
      result_dict 与原来 detect_middle_bar_projection 返回的 info 类似：
        {
          y_center, y_range, normalized_height, band_height,
          x_range, baseline_gap, used_gap_threshold, image_shape
        }
    """
    if gray_img is None or gray_img.size == 0:
        return None
    if len(gray_img.shape) != 2:
        return None

    # 二值（Otsu）
    _, th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
    bin_ = (th > 0).astype(np.uint8)

    # 膨胀
    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        for _ in range(dilate_iter):
            bin_ = cv2.dilate(bin_, kernel)

    h, w = bin_.shape
    if h < 3 or w < 3:
        return None

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

    # 连续分组
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
    h_norm = y_center / (h - 1) if h > 1 else 0.0

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
    return result, bin_

# ---- 4.b 细化 H / M 判定 ----
def refine_HM_label(char_roi,
                    init_label,
                    topk_scores,
                    debug_dir,
                    char_idx,
                    ratio_threshold=HM_BAR_RATIO_THRESHOLD,
                    score_close_delta=HM_SCORE_CLOSE_DELTA,
                    dilate_iter=HM_DILATE_ITER,
                    gap_drop_ratio=HM_GAP_DROP_RATIO):
    """
    根据中横条高度比例细化 H / M。
    返回:
      refined_label, extra_info(dict)
    """
    # 判断是否需要触发细化
    need_refine = False
    labels_in_topk = {l for l, _ in topk_scores}
    hm_in_topk = ('H' in labels_in_topk) and ('M' in labels_in_topk)

    if init_label in ('H', 'M'):
        need_refine = True
    if hm_in_topk:
        # 找出 H、M 分数
        scoreH = next((s for l, s in topk_scores if l == 'H'), None)
        scoreM = next((s for l, s in topk_scores if l == 'M'), None)
        if scoreH is not None and scoreM is not None and abs(scoreH - scoreM) < score_close_delta:
            need_refine = True

    if not need_refine:
        return init_label, {"refined": False}

    det = detect_middle_bar_projection_from_array(
        char_roi,
        dilate_iter=dilate_iter,
        gap_drop_ratio=gap_drop_ratio
    )
    if det is None:
        # 没检测到中横条：保持原模板结果
        return init_label, {
            "refined": True,
            "reason": "no_bar_detected",
            "band_ratio": None
        }

    info, bin_img = det
    band_height = info["band_height"]
    h_total = info["image_shape"][0]
    band_ratio = band_height / h_total if h_total > 0 else 0

    # 规则
    if band_ratio < ratio_threshold:
        final_label = 'H'
    else:
        final_label = 'M'

    # 可视化
    vis = cv2.cvtColor(char_roi, cv2.COLOR_GRAY2BGR)
    y_top, y_bottom = info["y_range"]
    x_min, x_max = info["x_range"]
    cv2.rectangle(vis, (x_min, y_top), (x_max, y_bottom), (0,0,255), 1)
    cv2.putText(vis,
                f"band_ratio={band_ratio:.2f}",
                (1, max(10, y_top-2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
    cv2.putText(vis,
                f"{init_label}->{final_label}",
                (1, min(h_total-2, y_bottom+10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    save_img(os.path.join(debug_dir, f"char{char_idx:02d}_HM_refine.png"), vis)

    return final_label, {
        "refined": True,
        "reason": "band_ratio_rule",
        "band_ratio": band_ratio,
        "band_height": band_height,
        "h_total": h_total
    }

# ---- 5. 主识别流程 ----
def recognize_image(image_path, template_dir, debug_dir="debug_outputs", min_area=30):
    ensure_dir(debug_dir)
    # step1: 读入原图
    raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    save_img(os.path.join(debug_dir, "01_raw.png"), raw)
    # step2: 二值化
    # _, bw = cv2.threshold(raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bw = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)

    save_img(os.path.join(debug_dir, "02_bw.png"), bw)

    # step2b: 全局腐蚀（可调参数）
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw_eroded = cv2.erode(bw, erode_kernel, iterations=1)
    save_img(os.path.join(debug_dir, "02b_bw_eroded.png"), bw_eroded)




    # step3: 分割字符区域
    chars = segment_characters(bw_eroded, min_area=min_area)
    # step4: 可视化分割结果
    vis_boxes = cv2.cvtColor(bw_eroded, cv2.COLOR_GRAY2BGR)
    for idx, c in enumerate(chars):
        x, y, w, h = c["bbox"]
        cv2.rectangle(vis_boxes, (x, y), (x+w, y+h), (0,0,255), 1)
        cv2.putText(vis_boxes, str(idx), (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    save_img(os.path.join(debug_dir, "03_seg_boxes.png"), vis_boxes)
    # step5: 加载模板
    templates = load_templates(template_dir)

    # step6: 循环匹配 + H/M 细化
    results = []
    roi_dir = os.path.join(debug_dir, "chars")
    ensure_dir(roi_dir)
    for idx, c in enumerate(chars):
        roi_path = os.path.join(roi_dir, f"char{idx:02d}_roi.png")
        save_img(roi_path, c["roi"])
        best_label, best_score, scores = match_char_to_templates(c["roi"], templates, roi_dir, idx)

        # 细化 H / M
        refined_label, hm_info = refine_HM_label(
            c["roi"],
            best_label,
            scores[:5],
            roi_dir,
            idx
        )

        results.append({
            "bbox": c["bbox"],
            "label_template": best_label,
            "label": refined_label,  # 使用最终标签
            "score": best_score,
            "topk": scores[:5],
            "hm_refine": hm_info
        })

    # step7: 结果可视化
    annotated = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    for idx, r in enumerate(results):
        x, y, w, h = r["bbox"]
        disp_label = r["label"]
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(annotated, f"{disp_label}:{r['score']:.2f}", (x, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    save_img(os.path.join(debug_dir, "04_annotated.png"), annotated)

    # step8: 结果文本
    recognized_str = "".join(r["label"] if r["label"] else "?" for r in results)
    with open(os.path.join(debug_dir, "final_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"recognized={recognized_str}\n")
        for idx, r in enumerate(results):
            f.write(
                f"{idx}\t{r['bbox']}\t{r['label_template']}->"
                f"{r['label']}\t{r['score']:.4f}\tHM={r['hm_refine']}\n"
            )
    print("识别结果：", recognized_str)
    return recognized_str, results

# ---- 用法示例 ----
if __name__ == "__main__":
    TEMPLATE_DIR = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\Letter\Muban_images"
    IMAGE_PATH = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\img_3.png"
    DEBUG_DIR = "debug_outputs_test5"
    recognize_image(IMAGE_PATH, TEMPLATE_DIR, DEBUG_DIR)
