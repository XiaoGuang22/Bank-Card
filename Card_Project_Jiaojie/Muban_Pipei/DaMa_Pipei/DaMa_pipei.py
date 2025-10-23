##########################这一个是银行卡账号的模板匹配识别的代码，主要是针对一个文件夹图片进行批量化识别的代码#############
#################其中账号是0到9数字字符，每个字符做了两个模板###################
###############建议代码：加入一个面积的筛选，防止一些小噪声也作为识别的对象###############
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

# ---- 1. 加载模板库 ----
def load_templates(template_dir):
    templates = {}
    for fname in os.listdir(template_dir):
        if not fname.lower().endswith('.png'):
            continue
        label = fname[0]
        img = cv2.imread(os.path.join(template_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None and label.isalnum():
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
        roi = bw_img[y:y + h, x:x + w]
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

def match_char_to_templates(char_roi, templates, debug_dir, char_idx):
    start_time = time.time()  # 计时开始
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
    elapsed_ms = int((time.time() - start_time) * 1000)  # 匹配耗时（毫秒）
    return best_label, best_score, scores, elapsed_ms

# ---- 4. 主识别流程 ----
def recognize_image(image_path, template_dir, debug_dir="debug_outputs", min_area=30):
    ensure_dir(debug_dir)
    raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    save_img(os.path.join(debug_dir, "01_raw.png"), raw)
    _, bw = cv2.threshold(raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_img(os.path.join(debug_dir, "02_bw.png"), bw)
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
        best_label, best_score, scores, elapsed_ms = match_char_to_templates(c["roi"], templates, roi_dir, idx)
        results.append({
            "bbox": c["bbox"],
            "label": best_label,
            "score": best_score,
            "topk": scores[:5],
            "match_elapsed_ms": elapsed_ms  # 匹配时间
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
    print("识别结果：", recognized_str)
    return recognized_str, results

# ---- 5. 批量识别+计时 ----
def batch_recognize_images(image_dir, template_dir, batch_debug_dir="debug_outputs_batch", min_area=30):
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

# ---- 6. 用法 ----
if __name__ == "__main__":
    # 修改为你自己的模板和图片文件夹路径
    #模板库的路径
    TEMPLATE_DIR = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\DaMa\DaMa_output"
    #识别图像的文件夹路径
    IMAGE_DIR = r"D:\PayCard_Detection\paycard_dectection_test1\Muban\DaMa\test_images"
    #输出识别过程图的路径
    BATCH_DEBUG_DIR = "debug_outputs_batch_test4"
    batch_recognize_images(IMAGE_DIR, TEMPLATE_DIR, BATCH_DEBUG_DIR)
