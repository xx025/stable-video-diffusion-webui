import cv2
from PIL import Image
from moviepy.editor import ImageSequenceClip
import imageio
import os
import numpy as np
import shutil


def image_pipeline_func(image_path, func_list):
    """
    :param image_path: 图片路径
    :param func_list: 图片处理函数列表
    :return:
    """
    image = Image.open(image_path)
    for func_dict in func_list:
        func = func_dict["func"]
        args = func_dict["args"]
        image = func(image, *args)
    return image


def enlarge_image(image, min_width=256, min_height=256, ):
    """
    如果尺寸小于指定值，则放大图片
    """
    # 获取原始图像的宽度和高度
    original_width, original_height = image.size
    # 判断是否需要调整大小
    if original_width >= min_width and original_height >= min_width:
        # 图像已经足够大，无需调整
        return image
    else:
        # 计算调整后的宽度和高度，保持宽高比例
        ratio = max(min_width / original_width, min_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        # 调整图像大小
        resized_image = image.resize((new_width, new_height))
        return resized_image


def shrink_image(image: Image, max_width=1024, max_height=1024):
    """
    如果尺寸大于指定值，则缩小图片
    """
    # 获取原始图像的宽度和高度
    original_width, original_height = image.size
    # 判断是否需要调整大小
    if original_width <= max_width and original_height <= max_height:
        # 图像已经足够小，无需调整
        return image
    else:
        # 计算调整后的宽度和高度，保持宽高比例
        ratio = min(max_width / original_width, max_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        # 调整图像大小
        resized_image = image.resize((new_width, new_height))
        return resized_image


def crop_to_nearest_multiple_of_n(image: Image, base=16) -> Image:
    # 获取原始图像的宽度和高度
    original_width, original_height = image.size
    # 判断是否需要进行变换
    if original_width % base == 0 and original_height % base == 0:
        # 图像的两边都已经是 16 的倍数，无需变换
        return image
    else:
        # 计算新的宽度和高度，分别为大于等于原始尺寸的 16 的倍数
        new_width = (original_width // base) * base
        new_height = (original_height // base) * base
        # 计算中心点坐标
        center_x, center_y = original_width // 2, original_height // 2
        half_new_width, half_new_height = new_width // 2, new_height // 2
        # 计算新的左上角坐标
        left = center_x - half_new_width
        top = center_y - half_new_height
        # 计算新的右下角坐标
        right = center_x + half_new_width
        bottom = center_y + half_new_height
        # 剪裁图像
        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image


def crop_center_resize(image: Image, target_width, target_height) -> Image:
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
    return resized_image


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

