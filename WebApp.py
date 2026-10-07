import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. إعداد الصفحة
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")
st.title("🌿 Plant Disease Detection Model")
st.write("ارفع صورة لورقة نبتة (مطيشة، بطاطا، أو فلفلة) لمعرفة ما إذا كانت مريضة أو سليمة.")

# 2. لائحة الأمراض (نفس الترتيب ديال التدريب)
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 
    'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# 3. دالة تحميل الموديل (كنستعملو st.cache_resource باش الموديل يتقرا مرة وحدة ومايتقالش التطبيق)
@st.cache_resource
def load_model():
    device = torch.device("cpu") # فـ Deployment غالبا كنخدمو بـ CPU
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 15)
    
    # تأكد من المسار ديال ملف الأوزان ديالك
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
    # عرض الصورة المرفوعة
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    st.write("جاري التحليل...")
    
    # 6. التوقع (Inference)
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        # استخراج النسبة المئوية للتأكد من ثقة الموديل
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    predicted_class = class_names[predicted_idx.item()]
    confidence_percentage = confidence.item() * 100
    
    # عرض النتيجة
    st.success(f"**النتيجة:** {predicted_class}")
    st.info(f"**نسبة الثقة (Confidence):** {confidence_percentage:.2f}%")