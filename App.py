import sys
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS 
import joblib
import time
import json
import concurrent.futures
import pandas as pd
import threading
import queue

# --- FIX LỖI PILLOW 10.0+ ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
    PIL.Image.linear_gradient = PIL.Image.new
# ---------------------------------------

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from geopy.geocoders import Nominatim
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel 

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==============================================================================
BASE_DIR = r"D:\LuanVan"
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models','Best_Medium_640_50e_10p_auto_20260119_1637.pt')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DET_PATH = os.path.join(MODEL_DIR, 'ch_PP-OCRv3_det_infer')
REC_PATH = os.path.join(MODEL_DIR, 'en_PP-OCRv4_rec_infer')
CLS_PATH = os.path.join(MODEL_DIR, 'ch_ppocr_mobile_v2.0_cls_infer')
VIETOCR_WEIGHT_PATH = os.path.join(MODEL_DIR, 'vietocr_FINAL_STAGE3_1103_0650.pth')
CLASSIFIER_PATH = os.path.join(BASE_DIR, "model_phanloai_danhmuc", "Naive_Bayes_model.pkl")

# [MODEL 0.5B]
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_ADAPTER_PATH = os.path.join(MODEL_DIR, 'qwen_0.5b_adapter_final_Step1')

# ==============================================================================
# CLASS ZOOMABLE CANVAS (Hỗ trợ Phóng to/Thu nhỏ & Kéo trượt ảnh)
# ==============================================================================
class ZoomableCanvas(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, bg="#333333", highlightthickness=0)
        self.hbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.cv_img = None
        self.pil_img = None
        self.tk_img = None
        self.zoom_scale = 1.0
        self.image_id = None

        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)

    def load_cv2_image(self, cv_img):
        self.cv_img = cv_img.copy()
        img_rgb = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
        self.pil_img = Image.fromarray(img_rgb)
        
        self.update_idletasks()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw > 10 and ch > 10:
            iw, ih = self.pil_img.size
            self.zoom_scale = min(cw/iw, ch/ih) * 0.95
        else:
            self.zoom_scale = 1.0
            
        self.show_image()

    def show_image(self):
        if self.pil_img is None: return
        w, h = int(self.pil_img.width * self.zoom_scale), int(self.pil_img.height * self.zoom_scale)
        if w < 10 or h < 10: return
        
        resized = self.pil_img.resize((w, h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        
        if self.image_id is not None:
            self.canvas.delete(self.image_id)
        
        self.image_id = self.canvas.create_image(max(self.canvas.winfo_width()//2, w//2), 
                                                 max(self.canvas.winfo_height()//2, h//2), 
                                                 anchor=tk.CENTER, image=self.tk_img)
        self.canvas.configure(scrollregion=self.canvas.bbox(tk.ALL))

    def on_mousewheel(self, event):
        if self.pil_img is None: return
        scale_factor = 1.1 if event.delta > 0 else 0.9
        self.zoom_scale *= scale_factor
        self.zoom_scale = max(0.1, min(self.zoom_scale, 10.0))
        self.show_image()

    def on_button_press(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def on_move_press(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def clear(self):
        self.canvas.delete("all")
        self.cv_img = None; self.pil_img = None; self.tk_img = None

# ==============================================================================
# LÕI XỬ LÝ BACKEND 
# ==============================================================================
class OCRBackend:
    def __init__(self):
        print("Đang tải models YOLO, OCR...")
        self.yolo = YOLO(YOLO_MODEL_PATH)
        self.paddle = PaddleOCR(det_model_dir=DET_PATH, rec_model_dir=REC_PATH, cls_model_dir=CLS_PATH,
                                det_limit_side_len=960, det_db_unclip_ratio=1.5, det_db_thresh=0.2,       
                                det_db_box_thresh=0.5, use_angle_cls=True, lang='en', show_log=False, rec=False, use_gpu=True)
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = VIETOCR_WEIGHT_PATH
        config['cnn']['pretrained'] = False
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.vietocr = Predictor(config)

        self.classifier = joblib.load(CLASSIFIER_PATH) if os.path.exists(CLASSIFIER_PATH) else None
        self.geolocator = Nominatim(user_agent="luan_van_app_v12")
        
        try:
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_ADAPTER_PATH)
            has_gpu = torch.cuda.is_available()
            base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16 if has_gpu else torch.float32,
                                                              device_map="auto" if has_gpu else None, offload_folder="offload_qwen")
            if not has_gpu: base_model = base_model.to("cpu")
            self.qwen_llm = PeftModel.from_pretrained(base_model, QWEN_ADAPTER_PATH)
        except Exception: self.qwen_llm = None

    # HÀM NẮN PHẲNG BIỂN HIỆU 3D CHUẨN
    def rectify_whole_sign(self, img):
        ocr_res = self.paddle.ocr(img, cls=False, det=True, rec=False)
        if not ocr_res or ocr_res[0] is None or len(ocr_res[0]) < 1:
            return img

        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
            return rect

        boxes = [order_points(np.array(b)) for b in ocr_res[0]]

        top_slopes, bot_slopes = [], []
        for b in boxes:
            if b[1][0] != b[0][0]: top_slopes.append((b[1][1] - b[0][1]) / (b[1][0] - b[0][0]))
            if b[2][0] != b[3][0]: bot_slopes.append((b[2][1] - b[3][1]) / (b[2][0] - b[3][0]))

        m_top = np.median(top_slopes) if top_slopes else 0
        m_bot = np.median(bot_slopes) if bot_slopes else 0

        all_pts = np.concatenate(boxes)
        pad_x = 15 
        min_x = np.min(all_pts[:, 0]) - pad_x
        max_x = np.max(all_pts[:, 0]) + pad_x

        pad_y = 15
        c_top = np.min(all_pts[:, 1] - m_top * all_pts[:, 0]) - pad_y
        c_bot = np.max(all_pts[:, 1] - m_bot * all_pts[:, 0]) + pad_y

        tl = [min_x, min_x * m_top + c_top]
        tr = [max_x, max_x * m_top + c_top]
        br = [max_x, max_x * m_bot + c_bot]
        bl = [min_x, min_x * m_bot + c_bot]

        if tl[1] >= bl[1] or tr[1] >= br[1]:
            return img

        src_pts = np.array([tl, tr, br, bl], dtype="float32")
        
        h_left = bl[1] - tl[1]
        h_right = br[1] - tr[1]
        h_new = int(max(h_left, h_right))
        w_new = int(max_x - min_x)

        ratio = max(h_left, h_right) / (min(h_left, h_right) + 1e-5)
        if 1.1 < ratio < 3.0:
            w_new = int(w_new * (1.0 + (ratio - 1.0) * 0.55)) 

        if w_new <= 0 or h_new <= 0: return img

        dst_pts = np.array([[0, 0], [w_new, 0], [w_new, h_new], [0, h_new]], dtype="float32")

        try:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img, M, (w_new, h_new), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return warped
        except Exception:
            return img

    # --- KHÔI PHỤC CÁC HÀM TIỀN XỬ LÝ & BẢO VỆ CHỮ ---
    def is_valid_text(self, text):
        if len(text.strip()) < 2: return False
        if not re.search(r'[a-zA-Z0-9]', text): return False
        return True

    def enhance_contrast(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def binary_threshold(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def sharpen_image(self, img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    def grayscale_eq(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    # --- CÁC HÀM XỬ LÝ DỮ LIỆU ---
    def get_decimal_from_dms(self, dms, ref):
        try:
            deg = float(dms[0]) + float(dms[1])/60.0 + float(dms[2])/3600.0
            return -deg if ref in ['S', 'W'] else deg
        except: return 0.0

    def get_clean_address(self, location):
        if not location: return "Không tìm thấy địa chỉ bản đồ"
        addr = location.raw.get('address', {})
        parts = []
        street_part = ""
        if 'house_number' in addr: street_part += f"Số {addr['house_number']}"
        if 'road' in addr:
            if street_part: street_part += " "
            street_part += addr['road']
        if street_part: parts.append(street_part)
        ward = addr.get('quarter') or addr.get('ward') or addr.get('village')
        if ward: parts.append(ward)
        district = addr.get('city_district') or addr.get('district') or addr.get('county') or addr.get('suburb')
        if district: parts.append(district)
        city = addr.get('city') or addr.get('state')
        if city: parts.append(city)
        return ", ".join(parts) if parts else location.address

    def get_geo_info(self, img_path):
        try:
            image = Image.open(img_path)
            exif_data = image._getexif()
            if not exif_data: return "Không có metadata", "Không xác định"
            gps = {GPSTAGS.get(k, k): v for t, val in exif_data.items() if TAGS.get(t, t) == "GPSInfo" for k, v in val.items()}
            if 'GPSLatitude' in gps:
                lat = self.get_decimal_from_dms(gps['GPSLatitude'], gps.get('GPSLatitudeRef', 'N'))
                lon = self.get_decimal_from_dms(gps['GPSLongitude'], gps.get('GPSLongitudeRef', 'E'))
                coords = f"{lat:.6f}, {lon:.6f}"
                try:
                    loc = self.geolocator.reverse(coords, timeout=5)
                    return coords, self.get_clean_address(loc)
                except: return coords, "Lỗi kết nối bản đồ"
            return "Không có GPS", "Không xác định"
        except: return "Lỗi", "Lỗi"

    def extract_info_and_clean_text(self, texts):
        combined = " ".join(texts)
        emails = list(set(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', combined)))
        webs = [w for w in re.findall(r'\b(?:www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}|[a-zA-Z0-9-]+\.(?:com|vn|net)(?:\.[a-zA-Z]{2,})?)\b', combined) if "@" not in w]
        return {"email": emails, "website": webs}

    def detect_robust(self, img):
        h, w = img.shape[:2]
        sf = min(1.0, 640 / max(h, w))
        img_s = cv2.resize(img, (int(w*sf), int(h*sf))) if sf < 1.0 else img
        res = self.paddle.ocr(img_s, cls=True, det=True, rec=False)
        return [np.array(b, dtype="float32") / sf for b in res[0]] if res and res[0] else []

    def fit_line_and_group(self, boxes):
        if not boxes: return []
        infos = [{'c': np.mean(b, axis=0), 'h': max(np.linalg.norm(b[0]-b[3]), np.linalg.norm(b[1]-b[2])), 'b': b} for b in boxes]
        infos.sort(key=lambda x: x['c'][0])
        used = set(); lines = []
        for i, s in enumerate(infos):
            if i in used: continue
            line = [s['b']]; used.add(i); last = s
            for j in range(i+1, len(infos)):
                if j in used: continue
                c = infos[j]
                if (min(c['h'], last['h'])/max(c['h'], last['h'])) < 0.5: continue
                if abs(c['c'][1]-last['c'][1]) > max(c['h'], last['h'])*0.8: continue
                if (c['c'][0]-last['c'][0]) > last['h']*5.0: continue
                line.append(c['b']); used.add(j); last = c
            lines.append(line)
        lines.sort(key=lambda ln: sum([np.mean(b, axis=0)[1] for b in ln])/len(ln))
        return lines

    def get_regression_rectified_crop(self, image, line_boxes):
        if not line_boxes: return None, None
        pts = np.concatenate(line_boxes, axis=0)
        tops = np.array([b[0] for b in line_boxes] + [b[1] for b in line_boxes], dtype=np.float32)
        bots = np.array([b[3] for b in line_boxes] + [b[2] for b in line_boxes], dtype=np.float32)
        [vxt, vyt, x0t, y0t] = cv2.fitLine(tops, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        [vxb, vyb, x0b, y0b] = cv2.fitLine(bots, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        minx, maxx = np.min(pts[:,0]), np.max(pts[:,0])
        
        def gy(x, vx, vy, x0, y0): return y0 if abs(vx)<1e-2 else y0+(vy/vx)*(x-x0)
        
        src = np.array([[minx, gy(minx, vxt, vyt, x0t, y0t)], [maxx, gy(maxx, vxt, vyt, x0t, y0t)],
                        [maxx, gy(maxx, vxb, vyb, x0b, y0b)], [minx, gy(minx, vxb, vyb, x0b, y0b)]], dtype="float32")
        wn = np.linalg.norm(src[1]-src[0]); hn = max(np.linalg.norm(src[3]-src[0]), np.linalg.norm(src[2]-src[1]))
        if hn>5000 or wn>5000 or hn<=0 or wn<=0: return None, None
        pw, ph = int(wn*0.05), int(hn*0.2)
        dst = np.array([[pw, ph], [wn+pw, ph], [wn+pw, hn+ph], [pw, hn+ph]], dtype="float32")
        try:
            return src.astype(int), cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (int(wn+pw*2), int(hn+ph*2)), flags=cv2.INTER_CUBIC, borderValue=(255,255,255))
        except: return None, None

    # --- LUỒNG XỬ LÝ CHÍNH ---
    def process_full_pipeline_generator(self, img_path):
        timings = {}; t_start_total = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_geo = executor.submit(self.get_geo_info, img_path)

            try:
                pil_image = ImageOps.exif_transpose(Image.open(img_path))
                frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception:
                frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            results_data = []
            
            # 1. YOLO
            t0 = time.time()
            yolo_res = self.yolo(frame, conf=0.5, verbose=False)
            timings['YOLO'] = time.time() - t0
            
            if not yolo_res[0].boxes: 
                yield {"step": "failed", "msg": "Không tìm thấy biển hiệu"}
                return
            
            box = yolo_res[0].boxes[0]; x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            yield {"step": "init", "original_frame": frame, "yolo_box": (x1, y1, x2, y2)}

            sign_crop = frame[y1:y2, x1:x2]
            sign_enhanced = self.enhance_contrast(sign_crop)
            
            # 2. Rectify 3D (Nắn biển hiệu)
            t0 = time.time()
            flat_sign = self.rectify_whole_sign(sign_enhanced)
            timings['Rectify_3D'] = time.time() - t0
            yield {"step": "yolo", "sign_crop": flat_sign, "time": timings['YOLO'] + timings['Rectify_3D']}

            # 3. Paddle (Tìm dòng)
            t0 = time.time()
            valid_boxes = [b for b in self.detect_robust(flat_sign) if (np.max(b[:,1])-np.min(b[:,1]) > max(12, int(flat_sign.shape[0]*0.04)))]
            lines = self.fit_line_and_group(valid_boxes)
            timings['PaddleOCR_Det'] = time.time() - t0
            yield {"step": "paddle", "time": timings['PaddleOCR_Det']}

            # 4. VietOCR (Nhận diện chữ & Lọc nhiễu)
            t0 = time.time()
            for idx, line_boxes in enumerate(lines):
                box_visual, crop_img = self.get_regression_rectified_crop(flat_sign, line_boxes)
                if crop_img is None or crop_img.size == 0: continue
                
                h, w = crop_img.shape[:2]; target_w = max(32, int(w * (64 / h)))
                crop_img = cv2.resize(crop_img, (target_w, 64), interpolation=cv2.INTER_CUBIC)

                best_candidate = {'prob': 0.0, 'text': '', 'name': '', 'img': crop_img}
                
                # KHÔI PHỤC BỘ LỌC ĐA LỚP
                filters = [("Original", lambda x: x), ("Sharpen", self.sharpen_image), 
                           ("Gray_Eq", self.grayscale_eq), ("Binary", self.binary_threshold)]
                           
                for fname, func in filters:
                    proc = func(crop_img)
                    pil_input = Image.fromarray(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
                    text, prob = self.vietocr.predict(pil_input, return_prob=True)
                    if prob > best_candidate['prob']:
                        best_candidate = {'name': fname, 'text': text, 'prob': prob, 'img': proc}
                    if prob > 0.88: break
                        
                # SỬ DỤNG HÀM LỌC CHỮ RÁC KHÔI PHỤC
                if best_candidate['prob'] > 0.8 and self.is_valid_text(best_candidate['text']):
                    results_data.append({
                        'id': idx + 1, 'box_points': box_visual.reshape((-1, 1, 2)),
                        'straight_img': crop_img, 'final_img': best_candidate['img'],
                        'filter_name': best_candidate['name'], 'text': best_candidate['text'], 'conf': best_candidate['prob']
                    })
                    
            timings['VietOCR_Rec'] = time.time() - t0
            yield {"step": "vietocr", "results_data": results_data, "time": timings['VietOCR_Rec']}

            # 5. LLM
            t0 = time.time()
            ext_json = {"BRAND":[],"SERVICE":[],"ADDRESS":[],"PHONE":[],"O":[]}
            if results_data and self.qwen_llm:
                ctx = "\n".join([f"[{int(np.min(r['box_points'][:,0,0]))}, {int(np.min(r['box_points'][:,0,1]))}, {int(np.max(r['box_points'][:,0,0]))}, {int(np.max(r['box_points'][:,0,1]))}] {r['text']}" for r in results_data])
                prompt = f"""Phân tích các dòng chữ OCR trích xuất vào JSON hợp lệ. Các trường: "BRAND" (Tên thương hiệu, ghép tên loại hình nếu đi kèm), "SERVICE" (Dịch vụ), "ADDRESS" (Địa chỉ), "PHONE" (Số điện thoại), "O" (Nhiễu/Slogan).
Đầu vào OCR:
{ctx}
Đầu ra JSON:"""
                inputs = {k: v.to(self.qwen_llm.device) for k, v in self.qwen_tokenizer([self.qwen_tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)], return_tensors="pt").items()}
                with torch.no_grad():
                    g_ids = self.qwen_llm.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=self.qwen_tokenizer.eos_token_id)
                res_text = self.qwen_tokenizer.batch_decode([g[len(i):] for i, g in zip(inputs["input_ids"], g_ids)], skip_special_tokens=True)[0]
                try:
                    ext_json = json.loads(res_text[res_text.find('{'):res_text.rfind('}')+1])
                except: pass
            timings['LLM_Qwen'] = time.time() - t0
            yield {"step": "llm", "extracted_json": ext_json, "time": timings['LLM_Qwen']}

            # 6. Classifier & Wrap up
            t0 = time.time()
            cat = "Chưa xác định"
            if results_data and self.classifier:
                br, sr = str(ext_json.get("BRAND","")), str(ext_json.get("SERVICE",""))
                cat = str(self.classifier.predict(pd.DataFrame([{'Brand': br, 'Service': sr}]))[0]) if br or sr else "Thiếu dữ liệu"
            timings['Classifier'] = time.time() - t0
            yield {"step": "classifier", "category": cat, "time": timings['Classifier']}

            gps, addr = future_geo.result()
            timings['Total'] = time.time() - t_start_total
            yield {"step": "done", "gps": gps, "geo_addr": addr, "info": self.extract_info_and_clean_text([r['text'] for r in results_data]), "timings": timings}


# ==============================================================================
# GIAO DIỆN NGƯỜI DÙNG (UI) - THEME MODERN
# ==============================================================================
class OCRInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Công cụ Thẩm định AI - Trích xuất Biển Hiệu (LuanVan v3)")
        self.root.geometry("1450x850")
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview", font=('Consolas', 11), rowheight=28)
        style.configure("Treeview.Heading", font=('Arial', 11, 'bold'))
        style.map('Treeview', background=[('selected', '#0078D7')], foreground=[('selected', 'white')])
        self.root.option_add("*TTreeview*highlightThickness", 0)

        self.backend = OCRBackend()
        self.current_results = []
        self.current_flat_sign = None

        self.build_ui()
        self.progress_queue = queue.Queue()

    def build_ui(self):
        top_frame = tk.Frame(self.root, bg="#2C3E50", height=60)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        top_frame.pack_propagate(False)
        
        title_lbl = tk.Label(top_frame, text="AI SIGNBOARD INSPECTOR", font=("Impact", 24), bg="#2C3E50", fg="white")
        title_lbl.pack(side=tk.LEFT, padx=20)
        
        self.lbl_status = tk.Label(top_frame, text="Sẵn sàng", font=("Arial", 12, "bold"), bg="#2C3E50", fg="#2ECC71")
        self.lbl_status.pack(side=tk.RIGHT, padx=20)

        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = tk.Frame(paned, width=500, bg="#ecf0f1")
        paned.add(left_frame, weight=1)

        self.btn_select = tk.Button(left_frame, text="CHỌN ẢNH XỬ LÝ", command=self.load_image_trigger, 
                                    font=("Arial", 12, "bold"), bg="#3498db", fg="white", height=2, relief=tk.FLAT, cursor="hand2")
        self.btn_select.pack(fill=tk.X, padx=10, pady=10)

        cat_frame = ttk.LabelFrame(left_frame, text=" Tổng Quan Biển Hiệu ")
        cat_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.lbl_category = tk.Label(cat_frame, text="LOẠI HÌNH: ...", font=("Arial", 14, "bold"), fg="#c0392b")
        self.lbl_category.pack(anchor="w", padx=10, pady=5)
        self.lbl_gps = tk.Label(cat_frame, text="📍 GPS: ...", font=("Arial", 10))
        self.lbl_gps.pack(anchor="w", padx=10)
        self.lbl_geo_addr = tk.Label(cat_frame, text="🗺 Vị trí: ...", font=("Arial", 10), wraplength=450)
        self.lbl_geo_addr.pack(anchor="w", padx=10, pady=(0,5))

        info_frame = ttk.LabelFrame(left_frame, text=" Dữ liệu Trích xuất (LLM) ")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        labels = [("🏢 Thương hiệu:", "lbl_brand", "#d35400"), 
                  ("🛠 Dịch vụ:", "lbl_service", "#2980b9"),
                  ("📞 Điện thoại:", "lbl_phone", "#16a085"),
                  ("🏠 Địa chỉ OCR:", "lbl_addr", "#8e44ad"),
                  ("✉ Liên hệ khác:", "lbl_contact", "black")]
        
        self.info_vars = {}
        for row, (text, var_name, color) in enumerate(labels):
            tk.Label(info_frame, text=text, font=("Arial", 10, "bold"), fg=color).grid(row=row, column=0, sticky="w", padx=10, pady=4)
            val_lbl = tk.Label(info_frame, text="...", font=("Arial", 10), wraplength=320, justify="left")
            val_lbl.grid(row=row, column=1, sticky="w", pady=4)
            self.info_vars[var_name] = val_lbl

        ocr_frame = ttk.LabelFrame(left_frame, text=" Danh sách Dòng chữ (Click để xem ảnh) ")
        ocr_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.tree = ttk.Treeview(ocr_frame, columns=("ID", "Conf", "Text"), show="headings")
        self.tree.heading("ID", text="#")
        self.tree.heading("Conf", text="Độ tin cậy")
        self.tree.heading("Text", text="Văn bản nhận diện")
        
        self.tree.column("ID", width=30, anchor="center")
        self.tree.column("Conf", width=80, anchor="center")
        self.tree.column("Text", width=350, anchor="w")
        
        tree_scroll_y = ttk.Scrollbar(ocr_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x = ttk.Scrollbar(ocr_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.tree.tag_configure('highlighted', background='#ffeaa7', font=('Consolas', 11, 'bold'))
        self.tree.bind('<ButtonRelease-1>', self.on_tree_select)

        right_frame = tk.Frame(paned, bg="white")
        paned.add(right_frame, weight=3)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_orig = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_orig, text=" [1] Ảnh Gốc & Khung YOLO ")
        self.canvas_orig = ZoomableCanvas(self.tab_orig)
        self.canvas_orig.pack(fill=tk.BOTH, expand=True)

        self.tab_rect = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_rect, text=" [2] Biển Hiệu (Nắn phẳng 3D) ")
        self.canvas_rect = ZoomableCanvas(self.tab_rect)
        self.canvas_rect.pack(fill=tk.BOTH, expand=True)

        # =========================================================
        # TAB 3: THIẾT KẾ LẠI GIAO DIỆN CHI TIẾT OCR (TẬN DỤNG KHOẢNG TRỐNG)
        # =========================================================
        self.tab_line = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_line, text=" [3] Trích xuất Dòng (OCR Input) ")
        
        lbl_instruct = tk.Label(self.tab_line, text="👈 Hãy nhấp vào một dòng trong bảng OCR bên trái để xem chi tiết ảnh cắt.", font=("Arial", 11, "italic"), fg="gray")
        lbl_instruct.pack(pady=10)
        
        img_frame = tk.Frame(self.tab_line)
        img_frame.pack(fill=tk.X, pady=5)
        
        self.lbl_crop_orig = tk.Label(img_frame, bg="#ecf0f1", relief=tk.SOLID, bd=1)
        self.lbl_crop_orig.pack(pady=(5,0))
        tk.Label(img_frame, text="Hình cắt thô", font=("Arial", 10)).pack(pady=(0,10))
        
        self.lbl_crop_proc = tk.Label(img_frame, bg="#ecf0f1", relief=tk.SOLID, bd=1)
        self.lbl_crop_proc.pack(pady=(5,0))
        tk.Label(img_frame, text="Hình đã qua Tiền xử lý (Đưa vào VietOCR)", font=("Arial", 10, "bold")).pack(pady=(0,10))
        
        detail_frame = ttk.LabelFrame(self.tab_line, text=" Phân tích chi tiết OCR ")
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        meta_frame = tk.Frame(detail_frame)
        meta_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.lbl_detail_conf = tk.Label(meta_frame, text="Độ tin cậy: ...", font=("Arial", 12, "bold"), fg="#27ae60")
        self.lbl_detail_conf.pack(side=tk.LEFT, padx=(0, 20))
        
        self.lbl_detail_filter = tk.Label(meta_frame, text="Bộ lọc sử dụng: ...", font=("Arial", 12, "bold"), fg="#2980b9")
        self.lbl_detail_filter.pack(side=tk.LEFT)
        
        tk.Label(detail_frame, text="Nội dung văn bản (TEXT):", font=("Arial", 11, "bold")).pack(anchor="w", padx=10, pady=(10,0))
        
        self.txt_detail_text = tk.Text(detail_frame, height=5, font=("Consolas", 14), wrap=tk.WORD, bg="#fdfbfb", relief=tk.SOLID, bd=1)
        self.txt_detail_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.txt_detail_text.config(state=tk.DISABLED)

    def format_val(self, val):
        if not val: return "Không tìm thấy"
        if isinstance(val, list): return " | ".join(val)
        return str(val)

    def load_image_trigger(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg;*.png;*.jpeg")])
        if not path: return
        
        self.btn_select.config(state=tk.DISABLED, bg="#95a5a6", text="ĐANG XỬ LÝ...")
        self.tree.delete(*self.tree.get_children())
        self.lbl_category.config(text="LOẠI HÌNH: Đang phân tích...")
        self.lbl_gps.config(text="📍 GPS: ...")
        self.lbl_geo_addr.config(text="🗺 Vị trí: ...")
        
        for k in self.info_vars: self.info_vars[k].config(text="...")
        
        self.canvas_orig.clear()
        self.canvas_rect.clear()
        self.lbl_crop_orig.config(image='')
        self.lbl_crop_proc.config(image='')
        
        # Xóa khung Text Tab 3
        self.txt_detail_text.config(state=tk.NORMAL)
        self.txt_detail_text.delete(1.0, tk.END)
        self.txt_detail_text.config(state=tk.DISABLED)
        self.lbl_detail_conf.config(text="Độ tin cậy: ...")
        self.lbl_detail_filter.config(text="Bộ lọc sử dụng: ...")
        
        self.notebook.select(0)
        
        threading.Thread(target=self.run_pipeline_worker, args=(path,), daemon=True).start()
        self.root.after(100, self.check_queue_and_update_ui)

    def run_pipeline_worker(self, path):
        try:
            for progress in self.backend.process_full_pipeline_generator(path):
                self.progress_queue.put(progress)
        except Exception as e:
            self.progress_queue.put({"step": "error", "msg": str(e)})

    def check_queue_and_update_ui(self):
        try:
            while not self.progress_queue.empty():
                progress = self.progress_queue.get_nowait()
                step = progress.get("step")
                
                if step == "error" or step == "failed":
                    self.lbl_status.config(text=f"LỖI: {progress.get('msg')}", fg="#e74c3c")
                    self.btn_select.config(state=tk.NORMAL, bg="#3498db", text="CHỌN ẢNH XỬ LÝ")
                    return
                
                elif step == "init":
                    orig = progress['original_frame'].copy()
                    x1, y1, x2, y2 = progress['yolo_box']
                    cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(orig, "YOLO Sign", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.canvas_orig.load_cv2_image(orig)
                    self.lbl_status.config(text="[1/5] Đang nắn phẳng phối cảnh...", fg="#f39c12")
                    
                elif step == "yolo":
                    self.current_flat_sign = progress["sign_crop"]
                    self.canvas_rect.load_cv2_image(self.current_flat_sign)
                    self.notebook.select(1)
                    self.lbl_status.config(text="[2/5] Đang gom khung dòng chữ...", fg="#f39c12")
                    
                elif step == "paddle":
                    self.lbl_status.config(text="[3/5] Đang đọc chữ bằng VietOCR...", fg="#f39c12")
                    
                elif step == "vietocr":
                    self.current_results = progress["results_data"]
                    for idx, res in enumerate(self.current_results):
                        self.tree.insert("", tk.END, iid=str(idx), values=(res['id'], f"{res['conf']:.2f}", res['text']))
                    self.lbl_status.config(text="[4/5] Đang bóc tách ý nghĩa (Qwen LLM)...", fg="#f39c12")
                    
                elif step == "llm":
                    js = progress["extracted_json"]
                    self.info_vars["lbl_brand"].config(text=self.format_val(js.get('BRAND')))
                    self.info_vars["lbl_service"].config(text=self.format_val(js.get('SERVICE')))
                    self.info_vars["lbl_phone"].config(text=self.format_val(js.get('PHONE')))
                    self.info_vars["lbl_addr"].config(text=self.format_val(js.get('ADDRESS')))
                    self.lbl_status.config(text="[5/5] Đang phân loại danh mục...", fg="#f39c12")
                    
                elif step == "classifier":
                    self.lbl_category.config(text=f"LOẠI HÌNH: {progress['category'].upper()}")
                    
                elif step == "done":
                    info = progress["info"]
                    self.lbl_gps.config(text=f"📍 GPS: {progress['gps']}")
                    self.lbl_geo_addr.config(text=f"🗺 Vị trí: {progress['geo_addr']}")
                    
                    contact_str = []
                    if info.get('email'): contact_str.append(f"Email: {self.format_val(info.get('email'))}")
                    if info.get('website'): contact_str.append(f"Web: {self.format_val(info.get('website'))}")
                    self.info_vars["lbl_contact"].config(text=" | ".join(contact_str) if contact_str else "Không tìm thấy")

                    self.lbl_status.config(text="✔ HOÀN TẤT XỬ LÝ!", fg="#2ecc71")
                    self.btn_select.config(state=tk.NORMAL, bg="#3498db", text="CHỌN ẢNH KHÁC")
                    return 
                    
        except queue.Empty: pass 
        self.root.after(100, self.check_queue_and_update_ui)

    def on_tree_select(self, event):
        selected_items = self.tree.selection()
        if not selected_items: return
        
        for item in self.tree.get_children():
            self.tree.item(item, tags=())
            
        item = selected_items[0]
        self.tree.item(item, tags=('highlighted',))
        
        idx = int(item)
        if idx >= len(self.current_results): return
        
        data = self.current_results[idx]
        
        viz_sign = self.current_flat_sign.copy()
        rect = cv2.minAreaRect(data['box_points'])
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(viz_sign, [box], 0, (0, 0, 255), 4) 
        self.canvas_rect.load_cv2_image(viz_sign)
        
        def get_tk_img(cv_img, target_h=100):
            h, w = cv_img.shape[:2]
            new_w = int(w * (target_h / h))
            img_rgb = cv2.cvtColor(cv2.resize(cv_img, (new_w, target_h)), cv2.COLOR_BGR2RGB)
            return ImageTk.PhotoImage(Image.fromarray(img_rgb))

        self.tk_crop_orig = get_tk_img(data['straight_img'], 80)
        self.lbl_crop_orig.config(image=self.tk_crop_orig)
        
        self.tk_crop_proc = get_tk_img(data['final_img'], 80)
        self.lbl_crop_proc.config(image=self.tk_crop_proc)
        
        # Cập nhật thông tin chi tiết vào Tab 3
        self.lbl_detail_conf.config(text=f"Độ tin cậy: {data['conf']:.2f}")
        self.lbl_detail_filter.config(text=f"Bộ lọc sử dụng: {data['filter_name']}")
        
        self.txt_detail_text.config(state=tk.NORMAL)
        self.txt_detail_text.delete(1.0, tk.END)
        self.txt_detail_text.insert(tk.END, data['text'])
        self.txt_detail_text.config(state=tk.DISABLED)

        self.notebook.select(2)

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRInspectorApp(root)
    root.mainloop()