from PIL import Image
from moviepy.editor import ImageSequenceClip
import imageio
import os
import numpy as np
import shutil

def auto_resize_image(input_path, output_path, max_width=1024, max_height=600):
    # 打开图片
    original_image = Image.open(input_path)
    # 获取原始图片的宽和高
    original_width, original_height = original_image.size

    if original_width <= max_width and original_height <= max_height:
        # 如果图片尺寸小于限定值，则直接保存原图
        original_image.save(output_path)
    else:
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





def create_video(input_images, output_video_path, frames_per_second=30):
    # Create a folder to store the frames
    output_folder = "output_frames"
    os.makedirs(output_folder, exist_ok=True)

    # Save the frames as images
    for i, frame in enumerate(input_images):
        frame_path = os.path.join(output_folder, f"frame_{i:06d}.png")
        imageio.imwrite(frame_path, np.array(frame))

    # Convert frames to video
    # 使用MoviePy将图像序列合成成视频
    clip = ImageSequenceClip(sorted([os.path.join(output_folder, frame) for frame in os.listdir(output_folder)]),
                             fps=frames_per_second)
    clip.write_videofile(output_video_path, codec="libx264")

    # 删除保存的图片帧
    shutil.rmtree(output_folder)

    return output_video_path
