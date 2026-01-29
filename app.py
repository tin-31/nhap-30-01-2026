# ... (Các phần import giữ nguyên)
# THÊM IMPORT MỚI
import tensorflow as tf
os.environ['SM_FRAMEWORK'] = 'tf.keras' # Cấu hình cho segmentation_models
import segmentation_models as sm 

# ... (Cấu hình trang giữ nguyên)

# CẬP NHẬT ID & PATH
SEG_FILE_ID = '1axOg7N5ssJrMec97eV-JMPzID26ynzN1' 
CLS_FILE_ID = '1-v64E5VqSvbuKDYtdGDJBqUcWe9QfPVe'

SEG_PATH = 'seg_model_new.keras'
CLS_PATH = 'TRUST_MED_CLS_BIRADS_FINAL.pth'

@st.cache_resource
def load_models():
    # Tải file
    if not os.path.exists(SEG_PATH):
        gdown.download(f'https://drive.google.com/uc?id={SEG_FILE_ID}', SEG_PATH, quiet=True)
    if not os.path.exists(CLS_PATH):
        gdown.download(f'https://drive.google.com/uc?id={CLS_FILE_ID}', CLS_PATH, quiet=True)

    # --- 1. LOAD KERAS MODEL (SEGMENTATION) ---
    with st.spinner("Đang khởi động TensorFlow... (Có thể hơi lâu)"):
        # Load model Keras
        seg_model = tf.keras.models.load_model(SEG_PATH, compile=False)
    
    # --- 2. LOAD PYTORCH MODEL (CLASSIFICATION) ---
    cls_model = models.efficientnet_b4(weights=None)
    cls_model.classifier[1] = torch.nn.Linear(cls_model.classifier[1].in_features, 4)
    cls_model.load_state_dict(torch.load(CLS_PATH, map_location='cpu'))
    cls_model.eval()
    
    return seg_model, cls_model

# ... (Phần validate_image giữ nguyên)

# SỬA LẠI PHẦN CHẠY MODEL TRONG NÚT UPLOAD
# ...
            # --- BƯỚC 1: PHÂN ĐOẠN (KERAS) ---
            # Keras cần input dạng: (1, 256, 256, 3) và giá trị 0-1 hoặc chuẩn hóa
            input_pil, nw, nh, dx, dy = letterbox_image(original_pil, (256, 256))
            input_np = np.array(input_pil) / 255.0 # Chuẩn hóa về 0-1
            input_tensor = np.expand_dims(input_np, axis=0) # Thêm batch dimension
            
            # Dự đoán bằng Keras
            mask_prob = seg_model.predict(input_tensor)[0,:,:,0] # Output: (256, 256)
            
            # Các bước xử lý mask giữ nguyên (cắt ROI...)
            mask_valid = mask_prob[dy:dy+nh, dx:dx+nw]
            # ... (Phần còn lại giữ nguyên logic cũ)

            # --- BƯỚC 2: PHÂN LOẠI (PYTORCH) ---
            # Chuyển ROI sang Tensor PyTorch
            # ... (Giữ nguyên code PyTorch cũ)
