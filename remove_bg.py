import os
from rembg import remove
from PIL import Image
from tqdm import tqdm

input_dir = "./PlantVillage"
output_dir = "./PlantVillage_NoBG"

os.makedirs(output_dir, exist_ok=True)

for root, _, files in os.walk(input_dir):
    for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            # مسار الصورة الأصلية
            input_path = os.path.join(root, file)
            
            # إنشاء نفس المجلدات في الدوسي الجديد
            relative_path = os.path.relpath(root, input_dir)
            save_folder = os.path.join(output_dir, relative_path)
            os.makedirs(save_folder, exist_ok=True)
            
            output_path = os.path.join(save_folder, file.replace('.jpg', '.png'))
            
            # إزالة الخلفية وحفظ الصورة
            try:
                input_img = Image.open(input_path)
                output_img = remove(input_img)
                # تحويل الخلفية الشفافة إلى لون أسود (أفضل للـ ResNet)
                background = Image.new("RGB", output_img.size, (0, 0, 0))
                background.paste(output_img, mask=output_img.split()[3]) 
                background.save(output_path)
            except Exception as e:
                print(f"Error processing {file}: {e}")