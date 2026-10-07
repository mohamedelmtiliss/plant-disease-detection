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

disease_info = {
    'Pepper__bell___Bacterial_spot': {
        'name': 'التبقع البكتيري (فلفل)',
        'cause': 'بكتيريا تنشط في الرطوبة العالية والحرارة المرتفعة.',
        'treatment': 'رش مبيدات زراعية تحتوي على النحاس، وتجنب السقي بالرش المباشر على الأوراق.'
    },
    'Pepper__bell___healthy': {
        'name': 'ورقة فلفل سليمة 🌿',
        'cause': 'النبتة بصحة جيدة.',
        'treatment': 'استمر في العناية الجيدة بالري والتسميد المعتدل.'
    },
    'Potato___Early_blight': {
        'name': 'اللفحة المبكرة (بطاطس)',
        'cause': 'فطر يتكاثر في درجات حرارة معتدلة مع رطوبة عالية.',
        'treatment': 'إزالة الأوراق السفلية المصابة، ورش مبيد فطري وقائي، وتجنب السقي الليلي.'
    },
    'Potato___Late_blight': {
        'name': 'اللفحة المتأخرة (بطاطس)',
        'cause': 'مرض فطري خطير جداً وسريع الانتشار في الجو البارد والرطب.',
        'treatment': 'تدمير النباتات المصابة فوراً لمنع العدوى، واستخدام مبيدات فطرية متخصصة (مثل مانكوزيب).'
    },
    'Potato___healthy': {
        'name': 'ورقة بطاطس سليمة 🌿',
        'cause': 'النبتة بصحة جيدة.',
        'treatment': 'مراقبة الحقل بانتظام للحفاظ على صحة المحصول.'
    },
    'Tomato_Bacterial_spot': {
        'name': 'التبقع البكتيري (طماطم)',
        'cause': 'بكتيريا تنتقل عبر البذور أو الماء المتطاير.',
        'treatment': 'الاعتماد على مبيدات النحاس، وإزالة الأوراق المصابة لتهوية النبتة.'
    },
    'Tomato_Early_blight': {
        'name': 'اللفحة المبكرة (طماطم)',
        'cause': 'فطر يصيب الأوراق القديمة أولاً في ظروف دافئة ورطبة.',
        'treatment': 'استخدام مبيدات فطرية، تقليم الأوراق القريبة من التربة، وتدوير المحاصيل.'
    },
    'Tomato_Late_blight': {
        'name': 'اللفحة المتأخرة (طماطم)',
        'cause': 'نفس الفطر المسبب للفحة البطاطس (Phytophthora infestans).',
        'treatment': 'رش مبيدات فطرية قوية فور ظهور العلامات الأولى والتخلص من بقايا النباتات.'
    },
    'Tomato_Leaf_Mold': {
        'name': 'عفن الأوراق (طماطم)',
        'cause': 'فطر ينمو في البيوت البلاستيكية بسبب قلة التهوية وارتفاع الرطوبة.',
        'treatment': 'تحسين التهوية داخل البيت البلاستيكي، تقليل الرطوبة، ورش مبيد فطري.'
    },
    'Tomato_Septoria_leaf_spot': {
        'name': 'تبقع الأوراق السبتوري (طماطم)',
        'cause': 'مرض فطري يسبب بقع دائرية صغيرة مع مركز رمادي.',
        'treatment': 'تجنب الري من الفوق، تنظيف الحقل من الأعشاب الضارة، ورش مبيد فطري وقائي.'
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'name': 'سوس العنكبوت (عث الطماطم)',
        'cause': 'حشرات دقيقة جداً تمتص عصارة النبات في الجو الحار والجاف.',
        'treatment': 'غسل الأوراق بالماء لتقليل الحشرات، واستخدام مبيدات العناكب (Acaricides) أو الزيوت النباتية (Neem Oil).'
    },
    'Tomato__Target_Spot': {
        'name': 'البقعة المستهدفة (طماطم)',
        'cause': 'فطر يسبب بقعاً تشبه لوحة التنشين.',
        'treatment': 'استخدام مبيدات فطرية واسعة المجال وتحسين دورة الهواء بين النباتات.'
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'name': 'فيروس تجعد واصفرار الأوراق (طماطم)',
        'cause': 'فيروس خطير تنقله حشرة "الذبابة البيضاء".',
        'treatment': 'لا يوجد علاج للفيروس. يجب اقتلاع النبتة المصابة وحرقها، ومكافحة الذبابة البيضاء بالمبيدات الحشرية لمنع انتشار العدوى.'
    },
    'Tomato__Tomato_mosaic_virus': {
        'name': 'فيروس تبرقش الطماطم (الموزاييك)',
        'cause': 'فيروس ينتقل عبر اللمس، الأدوات الزراعية، أو البذور.',
        'treatment': 'اقتلاع النباتات المصابة وتدميرها، تعقيم الأدوات الزراعية واليدين جيداً، ولا تزرع الطماطم في نفس المكان للعام القادم.'
    },
    'Tomato_healthy': {
        'name': 'ورقة طماطم سليمة 🌿',
        'cause': 'النبتة بصحة جيدة.',
        'treatment': 'حافظ على التسميد المتوازن والري المنتظم.'
    }
}

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
    
# استخراج معلومات المرض من الـ Dictionary
    disease_details = disease_info[predicted_class]
    
    # 8. عرض النتيجة النهائية بشكل احترافي
    st.markdown("---") # خط فاصل
    
    # يلا كانت النبتة سليمة غنعطيوها لون أخضر، ويلا مريضة لون أحمر/برتقالي
    if "healthy" in predicted_class:
        st.success(f"**النتيجة:** {disease_details['name']}")
    else:
        st.error(f"**المرض المتوقع:** {disease_details['name']}")
        
    st.info(f"**نسبة الثقة (Confidence):** {confidence_percentage:.2f}%")
    
    # 9. عرض الأسباب والنصائح في صندوق قابل للطي (Expander) باش تبقى الواجهة نقية
    with st.expander("🔍 اضغط هنا لمعرفة الأسباب وطرق العلاج", expanded=True):
        st.write(f"**🦠 المسبب / الأسباب:**")
        st.write(disease_details['cause'])
        st.write(f"**💉 العلاج والنصائح:**")
        st.write(disease_details['treatment'])