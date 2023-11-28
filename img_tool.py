from PIL import Image
import os
def auto_resize_image(input_path, output_path, max_width=1024, max_height=600):
    # 打开图片
    original_image = Image.open(input_path)
    # 获取原始图片的宽和高
    original_width, original_height = original_image.size
    # 计算缩放比例
    width_ratio = max_width / original_width
    height_ratio = max_height / original_height
    # 选择较小的比例进行缩放，以保持纵横比
    min_ratio = min(width_ratio, height_ratio)
    # 计算新的宽和高
    new_width = int(original_width * min_ratio)
    new_height = int(original_height * min_ratio)
    # 缩放图片
    resized_image = original_image.resize((new_width, new_height))
    # 保存缩放后的图片
    resized_image.save(output_path)

def crop_center(image_path, output_path, target_width, target_height):
    # 打开图像
    image = Image.open(image_path)
    # 计算裁剪的起始位置
    width, height = image.size
    aspect_ratio = target_width / target_height

    if width / height > aspect_ratio:  # 图像更宽
        new_width = int(aspect_ratio * height)
        offset = (width - new_width) / 2
        box = (offset, 0, width - offset, height)
    else:  # 图像更高
        new_height = int(width / aspect_ratio)
        offset = (height - new_height) / 2
        box = (0, offset, width, height - offset)
    # 中心裁剪到指定比例
    cropped_image = image.crop(box)
    # 调整裁剪后的图像尺寸
    resized_image = cropped_image.resize((target_width, target_height))
    # 保存裁剪并调整尺寸后的图像
    resized_image.save(output_path)

def generate_autocut_filename(original_filename):
    base_name, extension = os.path.splitext(original_filename)
    new_filename = f"{base_name}_autocut{extension}"
    return new_filename
