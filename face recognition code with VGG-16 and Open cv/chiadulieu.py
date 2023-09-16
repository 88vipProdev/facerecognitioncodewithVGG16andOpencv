import os
import shutil
import random

# Tạo thư mục gốc để lưu 2 tập huấn luyện là train và test
train_dir = os.path.join(os.getcwd(), 'dataset', 'train')
test_dir = os.path.join(os.getcwd(), 'dataset', 'test')
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Lấy danh sách tất cả các thư mục con trong thư mục 'image'
image_dir = os.path.join(os.getcwd(), 'E:/face recognition code with VGG-16 and Open cv/face recognition code with VGG-16 and Open cv/image')
all_folders = os.listdir(image_dir)

# Tạo thư mục con cho mỗi lớp và di chuyển ảnh vào tập train hoặc test
for folder in all_folders:
    class_name = folder
    src_folder = os.path.join(image_dir, folder)
    all_images = os.listdir(src_folder)
    for image in all_images:
        src = os.path.join(src_folder, image)
        if not os.path.exists(os.path.join(train_dir, class_name)):
            os.makedirs(os.path.join(train_dir, class_name))
        if not os.path.exists(os.path.join(test_dir, class_name)):
            os.makedirs(os.path.join(test_dir, class_name))
        # Random để tách dữ liệu thành 2 tập huấn luyện với tỷ lệ 80:20
        if random.random() < 0.8:
            dst = os.path.join(train_dir, class_name, image)
        else:
            dst = os.path.join(test_dir, class_name, image)
        shutil.copyfile(src, dst)

