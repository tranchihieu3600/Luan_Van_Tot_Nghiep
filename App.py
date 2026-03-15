import sys
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, Listbox, Scrollbar
from PIL import Image, ImageTk, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS 
import joblib
import time
import json

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

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==============================================================================
BASE_DIR = r"D:\LuanVan"
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models','Best_Medium_640_50e_10p_auto_20260119_1637.pt')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DET_PATH = os.path.join(MODEL_DIR, 'ch_PP-OCRv4_det_infer')
REC_PATH = os.path.join(MODEL_DIR, 'en_PP-OCRv4_rec_infer')
CLS_PATH = os.path.join(MODEL_DIR, 'ch_ppocr_mobile_v2.0_cls_infer')
VIETOCR_WEIGHT_PATH = os.path.join(MODEL_DIR, 'vietocr_FINAL_STAGE3_1103_0650.pth')
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "best_optimized_classifier.pkl")

# ---> ĐƯỜNG DẪN TỚI MODEL QWEN 0.5B <---
QWEN_MODEL_PATH = os.path.join(MODEL_DIR, 'qwen_0.5b_adapter_final')


class OCRBackend:
    def __init__(self):
        print("Đang tải models...")
        self.yolo = YOLO(YOLO_MODEL_PATH)
        
        self.paddle = PaddleOCR(
            det_model_dir=DET_PATH, rec_model_dir=REC_PATH,
            cls_model_dir=CLS_PATH,
            det_limit_side_len=2500, det_db_unclip_ratio=1.5, det_db_thresh=0.2,       
            det_db_box_thresh=0.5, use_angle_cls=True, lang='en', show_log=False, rec=False, use_gpu=True
        )
        
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = VIETOCR_WEIGHT_PATH
        config['cnn']['pretrained'] = False
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.vietocr = Predictor(config)

        self.classifier = None
        if os.path.exists(CLASSIFIER_PATH):
            try:
                self.classifier = joblib.load(CLASSIFIER_PATH)
                print(f"Đã load model phân loại.")
            except Exception as e:
                print(f"Lỗi load model phân loại: {e}")
        else:
            print(f"Không tìm thấy file model phân loại.")

        self.geolocator = Nominatim(user_agent="luan_van_app_v12")
        
        # Tải Model Qwen
        print("Đang tải model LLM Qwen 0.5b...")
        try:
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH)
            self.qwen_llm = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_PATH,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            print("Đã load Qwen thành công.")
        except Exception as e:
            print(f"Lỗi load model Qwen: {e}")
            self.qwen_llm = None

    def get_decimal_from_dms(self, dms, ref):
        try:
            degrees = float(dms[0])
            minutes = float(dms[1]) / 60.0
            seconds = float(dms[2]) / 3600.0
            decimal = degrees + minutes + seconds
            if ref in ['S', 'W']:
                decimal = -decimal
            return decimal
        except Exception:
            return 0.0

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
            gps_info = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "GPSInfo":
                    for key in value.keys():
                        sub_tag = GPSTAGS.get(key, key)
                        gps_info[sub_tag] = value[key]
            
            if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                lat = self.get_decimal_from_dms(gps_info['GPSLatitude'], gps_info.get('GPSLatitudeRef', 'N'))
                lon = self.get_decimal_from_dms(gps_info['GPSLongitude'], gps_info.get('GPSLongitudeRef', 'E'))
                coords_str = f"{lat:.6f}, {lon:.6f}"
                try:
                    location = self.geolocator.reverse(coords_str, timeout=5)
                    short_address = self.get_clean_address(location)
                except Exception as e:
                    short_address = "Lỗi kết nối bản đồ"
                return coords_str, short_address
            else:
                return "Không có GPS", "Không xác định"
        except Exception:
            return "Lỗi đọc file", "Lỗi"

    # --- ĐÃ SỬA: CHỈ CÒN LẠI REGEX EMAIL VÀ WEBSITE ---
    def extract_info_and_clean_text(self, detected_texts):
        info = {"email": [], "website": []}
        full_text_combined = " ".join(detected_texts)

        # Regex Email
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, full_text_combined)
        info["email"] = list(set(emails))

        # Regex Website (Các đuôi domain phổ biến)
        website_pattern = r'\b(?:www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}|[a-zA-Z0-9-]+\.(?:com|vn|net|org|edu|gov|info)(?:\.[a-zA-Z]{2,})?)\b'
        websites = re.findall(website_pattern, full_text_combined)
        
        # Lọc bỏ các email lỡ bị bắt nhầm vào website
        websites = [w for w in websites if w not in emails and "@" not in w]
        info["website"] = list(set(websites))

        return info

    def extract_information_with_llm(self, results_data):
        if not self.qwen_llm or not results_data:
            return {"BRAND": [], "SERVICE": [], "ADDRESS": [], "PHONE": [], "O": []}
        
        input_text = ""
        for res in results_data:
            pts = res['box_points'].reshape(-1, 2)
            xs = [int(pt[0]) for pt in pts]
            ys = [int(pt[1]) for pt in pts]
            xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
            text = res['text']
            input_text += f"[{xmin}, {ymin}, {xmax}, {ymax}] {text}\n"

        prompt = f"""Dưới đây là các văn bản được trích xuất từ một biển hiệu cùng với tọa độ của chúng [xmin, ymin, xmax, ymax].
Hãy trích xuất và phân loại các thông tin vào định dạng JSON với các trường sau:
- BRAND: Tên cửa hàng, tên thương hiệu.
- SERVICE: Dịch vụ, sản phẩm hoặc các thông tin liên quan tới cửa hàng.
- ADDRESS: Địa chỉ.
- PHONE: Số điện thoại.
- O: Chữ phụ, các thông tin không quan trọng.

Đầu vào:
{input_text.strip()}

Đầu ra JSON:"""

        messages = [{"role": "user", "content": prompt}]
        text_input = self.qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.qwen_tokenizer([text_input], return_tensors="pt").to(self.qwen_llm.device)

        # TỐI ƯU TỐC ĐỘ: max_new_tokens giảm xuống 256, do_sample=False
        generated_ids = self.qwen_llm.generate(
            **model_inputs, 
            max_new_tokens=256, 
            do_sample=False
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = self.qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        extracted_data = {"BRAND": [], "SERVICE": [], "ADDRESS": [], "PHONE": [], "O": []}
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                extracted_data = json.loads(json_str)
        except Exception as e:
            print("Lỗi parse JSON từ LLM:", e)

        return extracted_data

    # --- HÀM LỌC TEXT VÔ NGHĨA ---
    def is_valid_text(self, text):
        text = text.strip()
        if len(text) < 2: return False
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

    def detect_robust(self, img):
        h_orig, w_orig = img.shape[:2]
        scale_factor = 640 / max(h_orig, w_orig)
        if scale_factor >= 1.0: scale_factor = 1.0; img_small = img
        else:
            new_w = int(w_orig * scale_factor); new_h = int(h_orig * scale_factor)
            img_small = cv2.resize(img, (new_w, new_h))
        ocr_res = self.paddle.ocr(img_small, cls=True, det=True, rec=False)
        if not ocr_res or ocr_res[0] is None: return []
        final_boxes = []
        for box in ocr_res[0]:
            box_np = np.array(box, dtype="float32")
            if scale_factor < 1.0: box_np /= scale_factor 
            final_boxes.append(box_np)
        return final_boxes

    def fit_line_and_group(self, boxes):
        if not boxes: return []
        box_infos = []
        for i, b in enumerate(boxes):
            center = np.mean(b, axis=0)
            h = max(np.linalg.norm(b[0] - b[3]), np.linalg.norm(b[1] - b[2]))
            box_infos.append({'idx': i, 'c': center, 'h': h, 'box': b})
        box_infos.sort(key=lambda x: x['c'][0])
        used_indices = set()
        lines = []
        for i in range(len(box_infos)):
            if i in used_indices: continue
            seed = box_infos[i]
            current_line = [seed['box']]
            used_indices.add(i)
            last_box = seed 
            for j in range(i + 1, len(box_infos)):
                if j in used_indices: continue
                candidate = box_infos[j]
                max_h = max(candidate['h'], last_box['h'])
                min_h = min(candidate['h'], last_box['h'])
                if (min_h / max_h) < 0.5: continue 
                y_diff = abs(candidate['c'][1] - last_box['c'][1])
                if y_diff > max_h * 0.8: continue
                x_dist = candidate['c'][0] - last_box['c'][0]
                if x_dist > last_box['h'] * 5.0: continue
                current_line.append(candidate['box'])
                used_indices.add(j)
                last_box = candidate
            lines.append(current_line)
        lines.sort(key=lambda ln: sum([np.mean(b, axis=0)[1] for b in ln]) / len(ln))
        return lines

    def get_regression_rectified_crop(self, image, line_boxes):
        if not line_boxes: return None, None
        all_pts = np.concatenate(line_boxes, axis=0)
        tops = np.array([b[0] for b in line_boxes] + [b[1] for b in line_boxes], dtype=np.float32)
        bottoms = np.array([b[3] for b in line_boxes] + [b[2] for b in line_boxes], dtype=np.float32)
        line_top = cv2.fitLine(tops, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        line_bot = cv2.fitLine(bottoms, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        [vx_t, vy_t, x0_t, y0_t] = line_top; [vx_b, vy_b, x0_b, y0_b] = line_bot
        min_x = np.min(all_pts[:, 0]); max_x = np.max(all_pts[:, 0])

        def get_y_safe(x, vx, vy, x0, y0):
            return y0 if abs(vx) < 1e-2 else y0 + (vy/vx) * (x - x0)

        tl_y = get_y_safe(min_x, vx_t, vy_t, x0_t, y0_t); tr_y = get_y_safe(max_x, vx_t, vy_t, x0_t, y0_t)
        bl_y = get_y_safe(min_x, vx_b, vy_b, x0_b, y0_b); br_y = get_y_safe(max_x, vx_b, vy_b, x0_b, y0_b)
        src_pts = np.array([[min_x, tl_y], [max_x, tr_y], [max_x, br_y], [min_x, bl_y]], dtype="float32")

        w_new = np.linalg.norm(src_pts[1] - src_pts[0])
        h_new = max(np.linalg.norm(src_pts[3] - src_pts[0]), np.linalg.norm(src_pts[2] - src_pts[1]))
        if h_new > 5000 or w_new > 5000 or h_new <= 0 or w_new <= 0: return None, None
        pad_h = int(h_new * 0.2); pad_w = int(w_new * 0.05)
        dst_w = int(w_new + pad_w*2); dst_h = int(h_new + pad_h*2)
        dst_pts = np.array([[pad_w, pad_h], [dst_w-pad_w, pad_h], [dst_w-pad_w, dst_h-pad_h], [pad_w, dst_h-pad_h]], dtype="float32")
        try:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (dst_w, dst_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            return src_pts.astype(int), warped
        except: return None, None

    def process_full_pipeline(self, img_path):
        timings = {} 
        t_start_total = time.time()
        
        gps_coords, geo_address = self.get_geo_info(img_path)
        try:
            pil_image = Image.open(img_path)
            pil_image = ImageOps.exif_transpose(pil_image)
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception:
            stream = open(img_path, "rb"); bytes = bytearray(stream.read()); stream.close()
            frame = cv2.imdecode(np.asarray(bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        results_data = []
        
        # 1. Thời gian YOLO
        t_start_yolo = time.time()
        yolo_res = self.yolo(frame, conf=0.5, verbose=False)
        timings['YOLO'] = time.time() - t_start_yolo
        
        if not yolo_res[0].boxes: 
            timings['Total'] = time.time() - t_start_total
            return frame, [], "Không tìm thấy biển hiệu", {}, gps_coords, geo_address, {}, timings
        
        box = yolo_res[0].boxes[0]; x1, y1, x2, y2 = map(int, box.xyxy[0])
        sign_crop = frame[y1:y2, x1:x2]
        sign_enhanced = self.enhance_contrast(sign_crop)

        # 2. Thời gian PaddleOCR
        t_start_paddle = time.time()
        raw_boxes = self.detect_robust(sign_enhanced)
        valid_boxes = [b for b in raw_boxes if (np.max(b[:,1])-np.min(b[:,1]) > 8)]
        lines = self.fit_line_and_group(valid_boxes)
        timings['PaddleOCR_Det'] = time.time() - t_start_paddle

        # 3. Thời gian VietOCR
        t_start_vietocr = time.time()
        detected_texts = [] 
        for idx, line_boxes in enumerate(lines):
            try:
                box_visual, crop_img = self.get_regression_rectified_crop(sign_enhanced, line_boxes)
                if crop_img is None or crop_img.size == 0: continue

                h, w = crop_img.shape[:2]; target_h = 64; target_w = int(w * (target_h / h))
                if target_w <= 0: target_w = 32
                crop_img = cv2.resize(crop_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

                candidates = []
                for fname, func in [("Original", lambda x: x), 
                                    ("Binary", self.binary_threshold),
                                    ("Sharpen", self.sharpen_image),
                                    ("Gray_Eq", self.grayscale_eq)]:
                    proc = func(crop_img)
                    pil_input = Image.fromarray(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
                    text, prob = self.vietocr.predict(pil_input, return_prob=True)
                    candidates.append({'name': fname, 'text': text, 'prob': prob, 'img': proc})
                
                best = max(candidates, key=lambda x: x['prob'])
                
                if best['prob'] > 0.8 and self.is_valid_text(best['text']):
                    detected_texts.append(best['text'])
                    results_data.append({
                        'id': idx + 1,
                        'box_points': box_visual.reshape((-1, 1, 2)),
                        'straight_img': crop_img,
                        'final_img': best['img'],
                        'filter_name': best['name'],
                        'text': best['text'],
                        'conf': best['prob']
                    })
            except: pass
        timings['VietOCR_Rec'] = time.time() - t_start_vietocr

        # 4. Thời gian LLM Qwen bóc tách
        t_start_llm = time.time()
        extracted_json = self.extract_information_with_llm(results_data)
        timings['LLM_Qwen'] = time.time() - t_start_llm

        # 5. Regex thông tin mảng Email/Web
        detected_texts_list = [res['text'] for res in results_data] 
        extracted_info = self.extract_info_and_clean_text(detected_texts_list)

        # 6. Thời gian Phân loại
        t_start_classifier = time.time()
        predicted_category = "Chưa xác định"
        if results_data and self.classifier:
            try:
                brands = extracted_json.get("BRAND", [])
                services = extracted_json.get("SERVICE", [])
                if not isinstance(brands, list): brands = [brands]
                if not isinstance(services, list): services = [services]
                
                text_for_classification = " ".join(brands + services)
                
                if text_for_classification.strip():
                    predicted_category = str(self.classifier.predict([text_for_classification])[0])
                else:
                    predicted_category = "Chưa đủ dữ liệu (Không tìm thấy Brand/Service)"
            except Exception as e: 
                predicted_category = f"Lỗi Model: {e}"
        timings['Classifier'] = time.time() - t_start_classifier

        timings['Total'] = time.time() - t_start_total

        return sign_crop, results_data, predicted_category, extracted_info, gps_coords, geo_address, extracted_json, timings


class OCRInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Công cụ thử nghiệm OCR LuanVan")
        self.root.geometry("1400x800")
        
        self.backend = OCRBackend()
        self.current_sign_img = None
        self.current_results = []

        left_panel = tk.Frame(root, width=450, bg="#f0f0f0") 
        left_panel.pack_propagate(False) 
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        tk.Button(left_panel, text="Chọn Ảnh", command=self.load_image, 
                  height=2, bg="#4CAF50", fg="white", font=("Arial", 12, "bold")).pack(fill=tk.X, pady=5)
        
        self.lbl_category = tk.Label(left_panel, text="LOẠI HÌNH: ...", 
                                     font=("Arial", 16, "bold"), fg="#D32F2F", bg="#FFEBEE", pady=10)
        self.lbl_category.pack(fill=tk.X, pady=(5,5))

        # --- UPDATE UI PHẦN THÔNG TIN ---
        info_frame = tk.LabelFrame(left_panel, text="Thông tin trích xuất & Vị trí", font=("Arial", 10, "bold"), bg="white")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.lbl_gps = tk.Label(info_frame, text=" GPS: ...", anchor="w", justify="left", bg="white", fg="blue", wraplength=400)
        self.lbl_gps.pack(fill=tk.X, padx=5, pady=2)
        self.lbl_geo_addr = tk.Label(info_frame, text=" Vị trí ảnh: ...", anchor="w", justify="left", bg="white", fg="#00695C", wraplength=400, font=("Arial", 9, "bold"))
        self.lbl_geo_addr.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(info_frame, text="--- Thông tin trên biển hiệu ---", bg="white", fg="#888").pack(fill=tk.X, pady=2)
        
        # Thêm Brand, Service từ LLM lên giao diện chính
        self.lbl_brand = tk.Label(info_frame, text=" Thương hiệu: ...", anchor="w", justify="left", bg="white", wraplength=400, font=("Arial", 10, "bold"), fg="#D84315")
        self.lbl_brand.pack(fill=tk.X, padx=5, pady=2)
        self.lbl_service = tk.Label(info_frame, text=" Dịch vụ: ...", anchor="w", justify="left", bg="white", wraplength=400)
        self.lbl_service.pack(fill=tk.X, padx=5, pady=2)
        
        self.lbl_phone = tk.Label(info_frame, text=" SĐT: ...", anchor="w", justify="left", bg="white", wraplength=400, font=("Arial", 9, "bold"))
        self.lbl_phone.pack(fill=tk.X, padx=5, pady=2)
        self.lbl_addr = tk.Label(info_frame, text=" Địa chỉ: ...", anchor="w", justify="left", bg="white", wraplength=400)
        self.lbl_addr.pack(fill=tk.X, padx=5, pady=2)
        
        self.lbl_email = tk.Label(info_frame, text=" Email: ...", anchor="w", justify="left", bg="white")
        self.lbl_email.pack(fill=tk.X, padx=5, pady=2)
        self.lbl_website = tk.Label(info_frame, text=" Website: ...", anchor="w", justify="left", bg="white", fg="blue")
        self.lbl_website.pack(fill=tk.X, padx=5, pady=2)

        # Cập nhật tiêu đề Listbox
        tk.Label(left_panel, text="Danh sách dòng chữ & Thời gian xử lý:", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(pady=(10,0))
        list_frame = tk.Frame(left_panel)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        scrollbar = Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = Listbox(list_frame, font=("Consolas", 14), activestyle='none',
                               bg="white", fg="black", selectbackground="#0078D7", selectforeground="white",
                               yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind('<<ListboxSelect>>', self.on_select_line)

        right_panel = tk.Frame(root, bg="white")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(right_panel, text="[1] Vị trí trên biển hiệu", bg="white", fg="blue").pack(anchor="w")
        self.lbl_sign = tk.Label(right_panel, bg="#dddddd")
        self.lbl_sign.pack(pady=5)
        tk.Label(right_panel, text="[2] Sau khi cắt & xoay thẳng", bg="white", fg="blue").pack(anchor="w")
        self.lbl_straight = tk.Label(right_panel, bg="#dddddd")
        self.lbl_straight.pack(pady=5)
        tk.Label(right_panel, text="[3] Ảnh đã xử lý -> VietOCR Input", bg="white", fg="blue").pack(anchor="w")
        self.lbl_final = tk.Label(right_panel, bg="#dddddd")
        self.lbl_final.pack(pady=5)
        self.lbl_result = tk.Label(right_panel, text="KẾT QUẢ CHI TIẾT...", font=("Arial", 14, "bold"), fg="red", bg="white")
        self.lbl_result.pack(pady=10)

    def cv2_to_tk(self, cv_img, max_width=900, max_height=200):
        h, w = cv_img.shape[:2]
        scale = min(max_width/w, max_height/h)
        if scale < 1:
            new_w, new_h = int(w*scale), int(h*scale)
            cv_img = cv2.resize(cv_img, (new_w, new_h))
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(img_rgb))

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg;*.png;*.jpeg")])
        if not path: return
        
        self.listbox.delete(0, tk.END)
        self.lbl_result.config(text="Đang xử lý...")
        self.lbl_category.config(text="LOẠI HÌNH: Đang phân tích...")
        self.lbl_gps.config(text=" GPS: ...")
        self.lbl_geo_addr.config(text=" Vị trí ảnh: ...")
        
        # Clear Data
        self.lbl_brand.config(text=" Thương hiệu: ...")
        self.lbl_service.config(text=" Dịch vụ: ...")
        self.lbl_phone.config(text=" SĐT: ...")
        self.lbl_addr.config(text=" Địa chỉ: ...")
        self.lbl_email.config(text=" Email: ...")
        self.lbl_website.config(text=" Website: ...")
        self.root.update()

        try:
            sign_img, results, category, info, gps_coords, geo_addr, extracted_json, timings = self.backend.process_full_pipeline(path)
            
            self.current_sign_img = sign_img
            self.current_results = results

            self.tk_sign = self.cv2_to_tk(sign_img, max_height=300)
            self.lbl_sign.config(image=self.tk_sign)
            self.lbl_category.config(text=f"LOẠI HÌNH: {category.upper()}")
            self.lbl_gps.config(text=f" GPS: {gps_coords}")
            self.lbl_geo_addr.config(text=f" Vị trí ảnh: {geo_addr}")
            
            # --- FORMAT TEXT CHO GIAO DIỆN ---
            def format_val(val):
                if not val: return "Không tìm thấy"
                if isinstance(val, list): return ", ".join(val)
                return str(val)

            # Cập nhật thông tin từ LLM (Bỏ qua 'O')
            self.lbl_brand.config(text=f" Thương hiệu: {format_val(extracted_json.get('BRAND'))}")
            self.lbl_service.config(text=f" Dịch vụ: {format_val(extracted_json.get('SERVICE'))}")
            self.lbl_phone.config(text=f" SĐT: {format_val(extracted_json.get('PHONE'))}")
            self.lbl_addr.config(text=f" Địa chỉ:\n{format_val(extracted_json.get('ADDRESS'))}")
            
            # Cập nhật thông tin từ Regex
            self.lbl_email.config(text=f" Email: {format_val(info.get('email'))}")
            self.lbl_website.config(text=f" Website: {format_val(info.get('website'))}")

            # --- CẬP NHẬT LISTBOX (Đã xóa in JSON, chỉ in text và thời gian) ---
            if not results:
                self.listbox.insert(tk.END, "(Không tìm thấy chữ)")
            else:
                for res in results:
                    self.listbox.insert(tk.END, f"Dòng {res['id']}: {res['text']} ({res['conf']:.2f})")
                
                self.listbox.insert(tk.END, "")
                self.listbox.insert(tk.END, "=== THỜI GIAN XỬ LÝ ===")
                self.listbox.insert(tk.END, f"- YOLO (Cắt bảng): {timings.get('YOLO', 0):.2f}s")
                self.listbox.insert(tk.END, f"- PaddleOCR (Tìm khung): {timings.get('PaddleOCR_Det', 0):.2f}s")
                self.listbox.insert(tk.END, f"- VietOCR (Đọc chữ): {timings.get('VietOCR_Rec', 0):.2f}s")
                self.listbox.insert(tk.END, f"- LLM Qwen (Bóc tách): {timings.get('LLM_Qwen', 0):.2f}s")
                self.listbox.insert(tk.END, f"- Model Phân loại: {timings.get('Classifier', 0):.3f}s")
                self.listbox.insert(tk.END, f">> TỔNG CỘNG: {timings.get('Total', 0):.2f}s")

                self.lbl_result.config(text="Đã xử lý xong.")

        except Exception as e:
            messagebox.showerror("Lỗi", str(e))
            print(e)

    def on_select_line(self, event):
        selection = self.listbox.curselection()
        if not selection: return
        idx = selection[0]
        if idx >= len(self.current_results): return
        
        data = self.current_results[idx]
        viz_sign = self.current_sign_img.copy()
        rect = cv2.minAreaRect(data['box_points'])
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(viz_sign, [box], 0, (0, 255, 0), 3) 
        
        self.tk_sign_update = self.cv2_to_tk(viz_sign, max_height=300)
        self.lbl_sign.config(image=self.tk_sign_update)
        self.tk_straight = self.cv2_to_tk(data['straight_img'], max_height=100)
        self.lbl_straight.config(image=self.tk_straight)
        self.tk_final = self.cv2_to_tk(data['final_img'], max_height=100)
        self.lbl_final.config(image=self.tk_final)
        self.lbl_result.config(text=f"TEXT: {data['text']}\nĐộ tin cậy: {data['conf']:.2f} | Bộ lọc: {data['filter_name']}")


if __name__ == "__main__":
    root = tk.Tk()
    app = OCRInspectorApp(root)
    root.mainloop()