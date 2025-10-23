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
# def load_templates(template_dir):
#     templates = {}
#     for fname in os.listdir(template_dir):
#         if not fname.lower().endswith('.png'):
#             continue
#         # 提取第一个字符作为模板 key（支持所有字符，包括中文、标点等）
#         label = fname[0]
#         img = cv2.imread(os.path.join(template_dir, fname), cv2.IMREAD_GRAYSCALE)
#         if img is not None and label:  # label 非空即可
#             templates[label] = img
#     return templates
#
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
#
#     # step2: 二值化
#     _, bw = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
#
#
#
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
#     recognized_str = recognized_str.replace("_", "/")  # 替换 _ 为 /
#     with open(os.path.join(debug_dir, "final_result.txt"), "w", encoding="utf-8") as f:
#         f.write(f"recognized={recognized_str}\n")
#         for idx, r in enumerate(results):
#             f.write(f"{idx}\t{r['bbox']}\t{r['label']}\t{r['score']:.4f}\n")
#     print("识别结果：", recognized_str)
#     return recognized_str, results
#
# # ---- 用法 ----
# if __name__ == "__main__":
#     TEMPLATE_DIR = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\XiaoMa_number\xiaoma_muban_images"
#     IMAGE_PATH = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\XiaoMa_number\xiao_test_images\img_2.png"
#     DEBUG_DIR = "debug_outputs_test3"
#     recognize_image(IMAGE_PATH, TEMPLATE_DIR, DEBUG_DIR)




# import cv2
# import numpy as np
# import os
# import uuid
# import time
#
# def ensure_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#
# def save_img(path, img):
#     ensure_dir(os.path.dirname(path))
#     cv2.imwrite(path, img)
#
# def load_templates(template_dir):
#     templates = {}
#     for fname in os.listdir(template_dir):
#         if not fname.lower().endswith('.png'):
#             continue
#         label = fname[0]
#         img = cv2.imread(os.path.join(template_dir, fname), cv2.IMREAD_GRAYSCALE)
#         if img is not None and label:
#             templates[label] = img
#     return templates
#
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
#         char_resized = cv2.resize(char_roi, tpl_img.shape[::-1], interpolation=cv2.INTER_LINEAR)
#         score = ncc_score(char_resized, tpl_img)
#         scores.append((label, score))
#         save_img(os.path.join(debug_dir, f"char{char_idx:02d}_{label}_resized.png"), char_resized)
#         if score > best_score:
#             best_score = score
#             best_label = label
#     scores.sort(key=lambda x: x[1], reverse=True)
#     return best_label, best_score, scores
#
# def recognize_image(image_path, template_dir, debug_dir="debug_outputs", min_area=30):
#     ensure_dir(debug_dir)
#     raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     save_img(os.path.join(debug_dir, "01_raw.png"), raw)
#     _, bw = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
#     save_img(os.path.join(debug_dir, "02_bw.png"), bw)
#     chars = segment_characters(bw, min_area=min_area)
#     vis_boxes = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
#     for idx, c in enumerate(chars):
#         x, y, w, h = c["bbox"]
#         cv2.rectangle(vis_boxes, (x, y), (x+w, y+h), (0,0,255), 1)
#         cv2.putText(vis_boxes, str(idx), (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
#     save_img(os.path.join(debug_dir, "03_seg_boxes.png"), vis_boxes)
#     templates = load_templates(template_dir)
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
#     annotated = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
#     for idx, r in enumerate(results):
#         x, y, w, h = r["bbox"]
#         cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,0,255), 2)
#         cv2.putText(annotated, f"{r['label']}:{r['score']:.2f}", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
#     save_img(os.path.join(debug_dir, "04_annotated.png"), annotated)
#     recognized_str = "".join(r["label"] if r["label"] else "?" for r in results)
#     recognized_str = recognized_str.replace("_", "/")
#     with open(os.path.join(debug_dir, "final_result.txt"), "w", encoding="utf-8") as f:
#         f.write(f"recognized={recognized_str}\n")
#         for idx, r in enumerate(results):
#             f.write(f"{idx}\t{r['bbox']}\t{r['label']}\t{r['score']:.4f}\n")
#     print("识别结果：", recognized_str)
#     return recognized_str, results
#
# # ---- 批量识别并计时 ----
# def batch_recognize_images(image_dir, template_dir, batch_debug_dir="batch_debug_outputs", min_area=30):
#     ensure_dir(batch_debug_dir)
#     image_exts = ('.png', '.jpg', '.jpeg', '.bmp')
#     image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_exts)]
#     all_results_path = os.path.join(batch_debug_dir, "all_results.txt")
#     with open(all_results_path, "w", encoding="utf-8") as all_f:
#         for img_file in image_files:
#             img_path = os.path.join(image_dir, img_file)
#             debug_dir = os.path.join(batch_debug_dir, os.path.splitext(img_file)[0])
#             start_time = time.time()
#             recognized_str, results = recognize_image(img_path, template_dir, debug_dir, min_area=min_area)
#             end_time = time.time()
#             elapsed_ms = int((end_time - start_time) * 1000)
#             # 写入汇总结果
#             all_f.write(f"{img_file}\t{recognized_str}\t{elapsed_ms}ms\n")
#             for idx, r in enumerate(results):
#                 all_f.write(f"{img_file}\t{idx}\t{r['bbox']}\t{r['label']}\t{r['score']:.4f}\n")
#             print(f"{img_file} 识别完成，用时：{elapsed_ms} ms")
#     print("批量识别完成，总结果保存在：", all_results_path)
#
# # ---- 用法 ----
# if __name__ == "__main__":
#     # 修改为你自己的模板和图片文件夹路径
#     TEMPLATE_DIR = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\XiaoMa_number\xiaoma_muban_images"
#     IMAGE_DIR = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\XiaoMa_number\xiaoma_test_images"
#     BATCH_DEBUG_DIR = "batch_debug_outputs_test1"
#     batch_recognize_images(IMAGE_DIR, TEMPLATE_DIR, BATCH_DEBUG_DIR)




###################################这个代码说明了每个字符匹配的一个速度#################################
import cv2
import numpy as np
import os
import uuid
import time

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_img(path, img):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def load_templates(template_dir):
    templates = {}
    for fname in os.listdir(template_dir):
        if not fname.lower().endswith('.png'):
            continue
        label = fname[0]
        img = cv2.imread(os.path.join(template_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None and label:
            templates[label] = img
    return templates

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

def ncc_score(a, b):
    A = a.astype(np.float32)
    B = b.astype(np.float32)
    meanA, meanB = A.mean(), B.mean()
    num = np.sum((A - meanA) * (B - meanB))
    den = np.sqrt(np.sum((A - meanA) ** 2) * np.sum((B - meanB) ** 2))
    if den == 0:
        return -1.0
    return num / den

def match_char_to_templates(char_roi, templates, debug_dir, char_idx):
    start_time = time.time()  # 开始计时
    best_label = None
    best_score = -1.0
    scores = []
    for label, tpl_img in templates.items():
        char_resized = cv2.resize(char_roi, tpl_img.shape[::-1], interpolation=cv2.INTER_LINEAR)
        score = ncc_score(char_resized, tpl_img)
        scores.append((label, score))
        save_img(os.path.join(debug_dir, f"char{char_idx:02d}_{label}_resized.png"), char_resized)
        if score > best_score:
            best_score = score
            best_label = label
    scores.sort(key=lambda x: x[1], reverse=True)
    elapsed_ms = int((time.time() - start_time) * 1000)  # 计算耗时（毫秒）
    return best_label, best_score, scores, elapsed_ms

def recognize_image(image_path, template_dir, debug_dir="debug_outputs", min_area=30):
    ensure_dir(debug_dir)
    raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    save_img(os.path.join(debug_dir, "01_raw.png"), raw)
    _, bw = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
    save_img(os.path.join(debug_dir, "02_bw.png"), bw)
    chars = segment_characters(bw, min_area=min_area)
    vis_boxes = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for idx, c in enumerate(chars):
        x, y, w, h = c["bbox"]
        cv2.rectangle(vis_boxes, (x, y), (x+w, y+h), (0,0,255), 1)
        cv2.putText(vis_boxes, str(idx), (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    save_img(os.path.join(debug_dir, "03_seg_boxes.png"), vis_boxes)
    templates = load_templates(template_dir)
    results = []
    roi_dir = os.path.join(debug_dir, "chars")
    ensure_dir(roi_dir)
    for idx, c in enumerate(chars):
        roi_path = os.path.join(roi_dir, f"char{idx:02d}_roi.png")
        save_img(roi_path, c["roi"])
        best_label, best_score, scores, elapsed_ms = match_char_to_templates(c["roi"], templates, roi_dir, idx)
        results.append({
            "bbox": c["bbox"],
            "label": best_label,
            "score": best_score,
            "topk": scores[:5],
            "match_elapsed_ms": elapsed_ms  # 增加时间记录
        })
    annotated = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    for idx, r in enumerate(results):
        x, y, w, h = r["bbox"]
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(annotated, f"{r['label']}:{r['score']:.2f}", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    save_img(os.path.join(debug_dir, "04_annotated.png"), annotated)
    recognized_str = "".join(r["label"] if r["label"] else "?" for r in results)
    recognized_str = recognized_str.replace("_", "/")
    with open(os.path.join(debug_dir, "final_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"recognized={recognized_str}\n")
        for idx, r in enumerate(results):
            f.write(f"{idx}\t{r['bbox']}\t{r['label']}\t{r['score']:.4f}\t{r['match_elapsed_ms']}ms\n")
    print("识别结果：", recognized_str)
    return recognized_str, results

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
            # 写入汇总结果
            all_f.write(f"{img_file}\t{recognized_str}\t{elapsed_ms}ms\n")
            for idx, r in enumerate(results):
                all_f.write(f"{img_file}\t{idx}\t{r['bbox']}\t{r['label']}\t{r['score']:.4f}\t{r['match_elapsed_ms']}ms\n")
            print(f"{img_file} 识别完成，用时：{elapsed_ms} ms")
    print("批量识别完成，总结果保存在：", all_results_path)

# ---- 用法 ----
if __name__ == "__main__":
    # 修改为你自己的模板和图片文件夹路径
    TEMPLATE_DIR = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\XiaoMa_number\xiaoma_muban_images"
    IMAGE_DIR = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\XiaoMa_number\xiaoma_test_images"
    BATCH_DEBUG_DIR = "batch_debug_outputs_test4"
    batch_recognize_images(IMAGE_DIR, TEMPLATE_DIR, BATCH_DEBUG_DIR)

