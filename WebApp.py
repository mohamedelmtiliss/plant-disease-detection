import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# استيراد مكتبات Grad-CAM
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 1. إعداد الصفحة
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")
st.title("🌿 Plant Disease Detection with Explainable AI")
st.write("ارفع صورة لورقة نبتة لمعرفة المرض ورؤية المناطق المصابة (Grad-CAM Heatmap).")

# 2. لائحة الأمراض
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 
    'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# 3. دالة تحميل الموديل
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 15)
    
    model.load_state_dict(torch.load('./Models/best_plant_resnet50.pth', map_location=device))
    model.eval()
    return model

model = load_model()

# 4. إعداد الـ Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. واجهة رفع الصور
uploaded_file = st.file_uploader("اختر صورة للورقة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # تقسيم الشاشة لجوج أعمدة باش يبان داكشي متناسق
    col1, col2 = st.columns(2)
    
    # الصورة الأصلية
    image = Image.open(uploaded_file).convert('RGB')
    with col1:
        st.subheader("الصورة الأصلية")
        st.image(image, use_container_width=True)
    
    st.write("جاري التحليل...")
    
    # 6. التوقع (Inference)
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    predicted_class = class_names[predicted_idx.item()]
    confidence_percentage = confidence.item() * 100
    
    # 7. تطبيق GradCAM++ لتفاصيل أدق
    # نختارو layer3 بوحدها حيت كتشد التفاصيل الصغيرة (البقع)
    target_layers = [model.layer3[-1]]
    targets = [ClassifierOutputTarget(predicted_idx.item())]
    
    # استعمال GradCAM++
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    
    # حيدنا aug_smooth و eigen_smooth باش ما يتخلطوش لينا البقع وتولي ضبابة
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
    
    # تصفية قوية: نخليو غير البلايص اللي الموديل متأكد منهم بنسبة 40% الفوق
    threshold = 0.6
    grayscale_cam[grayscale_cam < threshold] = 0
    
    # تلوين الخريطة الحرارية
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    orig_img = np.array(image.resize((224, 224)))
    
    # دمج الألوان بوضوح
    blended = cv2.addWeighted(heatmap, 0.6, orig_img, 0.4, 0)
    
    # القناع (Mask)
    mask = (grayscale_cam > 0).astype(np.uint8)[:, :, np.newaxis]
    final_visualization = blended * mask + orig_img * (1 - mask)
    
    # عرض الـ Heatmap
    with col2:
        st.subheader("أماكن تركيز الموديل (GradCAM++)")
        st.image(final_visualization, use_container_width=True)
    
    # 8. عرض النتيجة النهائية
    st.markdown("---") # خط فاصل
    st.success(f"**المرض المتوقع:** {predicted_class}")
    st.info(f"**نسبة الثقة (Confidence):** {confidence_percentage:.2f}%")