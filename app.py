# ==========================================
# 🩺 TRUST-MED AI: HỆ THỐNG HỖ TRỢ CHẨN ĐOÁN UNG THƯ VÚ
# ==========================================
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import models, transforms
import os
import gdown
import matplotlib.pyplot as plt
import pandas as pd
import time

# =====================================================
# ⚙️ CẤU HÌNH GIAO DIỆN CHUNG
# =====================================================
st.set_page_config(
    page_title="TRUST-MED AI Support System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', sans-serif; }
    .stAlert { border-radius: 8px; }
    .report-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #0066cc;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================
# 1. CẤU HÌNH & TẢI MODEL
# ============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🔥 CẬP NHẬT ID MODEL MỚI THEO YÊU CẦU CỦA BẠN
# Lưu ý: Đây là file .keras, code sẽ cố gắng load nhưng cần file .pth để chạy đúng
SEG_FILE_ID = '1axOg7N5ssJrMec97eV-JMPzID26ynzN1' 
CLS_FILE_ID = '1-v64E5VqSvbuKDYtdGDJBqUcWe9QfPVe'

# Tên file lưu tạm trên máy chủ
SEG_PATH = 'seg_model_new.keras' # Đuôi .keras để gdown không bị lỗi định dạng
CLS_PATH = 'TRUST_MED_CLS_BIRADS_FINAL.pth'

@st.cache_resource
def load_models():
    # Tải file từ Drive
    if not os.path.exists(SEG_PATH):
        with st.spinner("📥 Đang tải Model Phân đoạn (Segmentation)..."):
            gdown.download(f'https://drive.google.com/uc?id={SEG_FILE_ID}', SEG_PATH, quiet=True)
            
    if not os.path.exists(CLS_PATH):
        with st.spinner("📥 Đang tải Model Phân loại (Classification)..."):
            gdown.download(f'https://drive.google.com/uc?id={CLS_FILE_ID}', CLS_PATH, quiet=True)

    # --- 1.1 SETUP SEGMENTATION MODEL ---
    # Cảnh báo: Kiến trúc U-Net ResNet34
    seg_model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, decoder_attention_type="scse")
    
    # 🛡️ CƠ CHẾ BẮT LỖI ĐỊNH DẠNG (Quan trọng cho trường hợp của bạn)
    try:
        # Cố gắng load weight vào model PyTorch
        seg_model.load_state_dict(torch.load(SEG_PATH, map_location=torch.device(DEVICE)))
    except RuntimeError as e:
        # Lỗi lệch kiến trúc hoặc file không phải PyTorch
        if "metadata.json" in str(e) or "zip" in str(e) or "magic number" in str(e):
            st.error(f"""
            ❌ **LỖI ĐỊNH DẠNG MODEL:** File `{SEG_PATH}` bạn cung cấp là định dạng **Keras/TensorFlow**, không chạy được trên code **PyTorch**.
            
            👉 **Cách sửa:** Hãy tìm file model có đuôi `.pth` hoặc `.pt` và cập nhật lại ID.
            """)
            st.stop() # Dừng app lại an toàn
        else:
            st.error(f"❌ Lỗi kiến trúc model (Key Mismatch): Model bạn tải lên có thể không phải là ResNet34+Unet+SCSE. Chi tiết: {e}")
            st.stop()
    except Exception as e:
        st.error(f"❌ Lỗi không xác định khi load Model Phân đoạn: {e}")
        st.stop()
        
    seg_model.to(DEVICE)
    seg_model.eval()
    
    # --- 1.2 SETUP CLASSIFICATION MODEL ---
    cls_model = models.efficientnet_b4(weights=None)
    cls_model.classifier[1] = torch.nn.Linear(cls_model.classifier[1].in_features, 4)
    try:
        cls_model.load_state_dict(torch.load(CLS_PATH, map_location=torch.device(DEVICE)))
    except Exception as e:
        st.error(f"❌ Lỗi load Model Phân loại: {e}")
        st.stop()
        
    cls_model.to(DEVICE)
    cls_model.eval()
    
    return seg_model, cls_model

# Load model
try:
    seg_model, cls_model = load_models()
except Exception as e:
    st.error(f"Hệ thống dừng do lỗi nạp model.")
    st.stop()

# ============================
# 2. CÁC HÀM XỬ LÝ ẢNH
# ============================
def validate_image(image_pil):
    img_np = np.array(image_pil)
    if img_np.shape[0] < 100 or img_np.shape[1] < 100: return False, "Kích thước quá nhỏ"
    if len(img_np.shape) == 3 and np.std(img_np, axis=2).mean() > 20: 
        return False, "Ảnh màu (không phải siêu âm)"
    return True, "Hợp lệ"

def letterbox_image(image, size):
    iw, ih = image.size; w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale); nh = int(ih*scale)
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image, nw, nh, (w-nw)//2, (h-nh)//2

def post_process_mask(mask_prob, threshold=0.5):
    mask_binary = (mask_prob > threshold).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    if num > 1:
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_clean = np.zeros_like(mask_binary)
        mask_clean[labels == max_label] = 1
        return mask_clean
    return mask_binary

def get_bounding_box(mask_pred, padding=0.2):
    cnts, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        pad_w = int(w*padding); pad_h = int(h*padding)
        x1 = max(0, x-pad_w); y1 = max(0, y-pad_h)
        x2 = min(mask_pred.shape[1], x+w+pad_w)
        y2 = min(mask_pred.shape[0], y+h+pad_h)
        return (x1, y1, x2, y2), "ROI"
    return (0,0,0,0), "None"

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.target_layer = target_layer
        self.gradients = None; self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]
    def __call__(self, x):
        output = self.model(x); idx = torch.argmax(output)
        self.model.zero_grad(); output[0, idx].backward()
        grads = self.gradients[0]; acts = self.activations[0]
        weights = torch.mean(grads, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * acts, dim=0).cpu().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - np.min(cam)) / (np.max(cam) + 1e-8)
        return cam, int(idx), torch.nn.functional.softmax(output, dim=1)

cam_extractor = GradCAM(cls_model, cls_model.features[-1])

def calc_trust_score(probs, mask_area_ratio):
    probs_np = probs.detach().cpu().numpy()[0]
    entropy = -np.sum(probs_np * np.log(probs_np + 1e-9))
    score_cls = 1.0 - (entropy / np.log(4))
    score_seg = 0.3 if mask_area_ratio < 0.01 else 0.95
    return 0.7 * score_cls + 0.3 * score_seg

# =====================================================
# UI LAYOUT
# =====================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
st.sidebar.title("TRUST-MED AI")
st.sidebar.markdown("**Hệ thống hỗ trợ chẩn đoán hình ảnh**")
menu = st.sidebar.radio("Chức năng:", ["🏠 Chẩn đoán", "📖 Hướng dẫn", "ℹ️ Giới thiệu"])

if menu == "🏠 Chẩn đoán":
    st.title("🖥️ Bàn làm việc Bác sĩ")
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("📥 Nhập dữ liệu")
        uploaded_file = st.file_uploader("Chọn ảnh siêu âm", type=["jpg", "png", "jpeg"])
        with st.expander("⚙️ Cấu hình"):
            seg_threshold = st.slider("Độ nhạy", 0.1, 0.9, 0.5)
            use_post_process = st.checkbox("Khử nhiễu", value=True)

    with col_right:
        if uploaded_file:
            original_pil = Image.open(uploaded_file).convert("RGB")
            original_np = np.array(original_pil)
            is_valid, msg = validate_image(original_pil)
            
            if not is_valid:
                st.error(f"⛔️ {msg}")
            else:
                progress = st.progress(0, "Đang xử lý...")
                
                # Segmentation
                progress.progress(30, "Đang phân đoạn...")
                input_pil, nw, nh, dx, dy = letterbox_image(original_pil, (256, 256))
                trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                inp_tensor = trans(input_pil).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    mask_prob = torch.sigmoid(seg_model(inp_tensor)).cpu().numpy()[0,0]
                
                mask_valid = mask_prob[dy:dy+nh, dx:dx+nw]
                mask_resized = cv2.resize(mask_valid, (original_np.shape[1], original_np.shape[0]))
                mask_binary = post_process_mask(mask_resized, seg_threshold) if use_post_process else (mask_resized > seg_threshold).astype(np.uint8)

                # Classification
                progress.progress(60, "Đang phân loại...")
                (x1, y1, x2, y2), _ = get_bounding_box(mask_binary)
                roi_img = original_np[y1:y2, x1:x2]
                
                roi_pil = Image.fromarray(roi_img)
                inp_cls = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(roi_pil).unsqueeze(0).to(DEVICE)
                
                heatmap, _, probs = cam_extractor(inp_cls)
                mask_ratio = np.sum(mask_binary) / (original_np.shape[0]*original_np.shape[1])
                trust_score = calc_trust_score(probs, mask_ratio)
                
                probs_np = probs.detach().cpu().numpy()[0]
                prob_mal = probs_np[2] + probs_np[3]
                prob_ben = probs_np[0] + probs_np[1]
                
                progress.progress(100, "Hoàn tất!"); time.sleep(0.5); progress.empty()

                # Report
                color = "green" if mask_ratio < 0.005 else ("red" if prob_mal > prob_ben else "blue")
                status = "BI-RADS 1" if mask_ratio < 0.005 else ("NGHI NGỜ ÁC TÍNH" if prob_mal > prob_ben else "LÀNH TÍNH")
                
                st.markdown(f"""
                <div class="report-box">
                    <h3 style="color:{color}; margin:0;">📋 KẾT QUẢ: {status}</h3>
                    <p>Độ tin cậy: {trust_score:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Lành tính", f"{prob_ben:.1%}")
                c2.metric("Ác tính", f"{prob_mal:.1%}")
                c3.metric("Kích thước u", f"{mask_ratio*100:.2f}%")
                
                st.divider()
                t1, t2, t3 = st.tabs(["Ảnh gốc", "Tổn thương", "AI Heatmap"])
                t1.image(original_pil, use_column_width=True)
                
                mask_disp = original_np.copy()
                mask_disp[mask_binary==1] = [0,255,0]
                t2.image(cv2.addWeighted(original_np,0.7,mask_disp,0.3,0), use_column_width=True)
                
                hm_color = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                t3.image(cv2.addWeighted(cv2.resize(roi_img,(224,224)),0.6,hm_color,0.4,0), use_column_width=True)

elif menu == "📖 Hướng dẫn":
    st.title("Hướng dẫn sử dụng")
    st.markdown("Tải ảnh siêu âm lên để hệ thống tự động phân tích.")

elif menu == "ℹ️ Giới thiệu":
    st.title("Giới thiệu")
    st.markdown("Hệ thống TRUST-MED AI hỗ trợ chẩn đoán ung thư vú.")
