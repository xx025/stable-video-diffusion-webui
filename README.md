# Stable-Video-Diffusion-WebUI


[English](README.md) | [中文](README_zh.md)






|![b48a6c775c944f7d89043ab8e0154197](https://github.com/xx025/stable-video-diffusion-webui/assets/71559822/0fa07557-10b8-4e64-b287-13d246170fc9) | ![000009](https://github.com/xx025/stable-video-diffusion-webui/assets/71559822/7984eb26-ae83-4f22-8e71-5cda2e8be708)|
|---------------------------------|----------------------------------|
| img                     | videos                      |



## Preparation

```shell
git clone https://github.com/xx025/stable-video-diffusion-webui.git
cd stable-video-diffusion-webui
```

Before starting, you need to create a virtual environment. It is recommended to use conda. If you are not comfortable using conda, create a Python 3.10 environment in your own way.

```shell
conda create -n svd python=3.10 
conda activate svd
```

Then execute the following commands in order,

```shell
git clone https://github.com/Stability-AI/generative-models.git

# During this process, the torch version corresponding to cu118 will be installed. You can try installing it first. The author tested it on a machine with cuda11.6, and it installed normally. 
# If it doesn't fit your computer, please change the --extra-index-url in requirements.txt accordingly.
python install.py

# Create a folder to save weight files
mkdir checkpoints 
# Download the model. This may take some time.
# If it's inconvenient to use wget, you can download it manually and save it to the checkpoints folder with the name svd_xt.safetensors
wget https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors?download=true -P checkpoints/ -O svd_xt.safetensors
```

## Run

During the first run, other model files will be downloaded.

```shell
python img2vid.py
```
If you wish to set a password, please go to seeting.py to make the changes.

## Personalized Settings

Please go to `setting.py` to make changes.
```python
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Specify the NVIDIA GPU to use

GR_SHARE = True
# generate a gradio share link, you can access it through the gradio website

auth = dict(
    # USER_NAME='root',
    # PASSWORD='123456',
    AUTH_MESSAGE='Place enter your username and password; 请输入用户名何密码',
)
# set username and password， if you don't want to set it, please comment it out

auto_adjust_img = dict(
    min_width=256,  # 图片最小宽度 Image minimum width
    min_height=256,  # 图片最小高度 Image minimum height
    max_height=1024,  # 图片最大宽度 Image maximum width
    max_width=1024,  # 图片最大高度 Image maximum height
    multiple_of_N=16  # 图片的宽高必须是N的倍数 The width and height of the image must be a multiple of N
)
# Automatically adjust the image resolution, automatically adjust to the resolution that meets the requirements


img_resize_to_HW = dict(
    target_width=1024,  # 目标宽度
    target_height=576,  # 目标高度
)
# Because often, using images at training size can achieve better
# results, but hard cropping can distort the images, so we use center cropping.

creat_video_by_opencv = False
# Use opencv to generate video, but it is found that there will be some encoding problems,
# so it is turned off by default,default use moviepy
```
![image](https://github.com/xx025/stable-video-diffusion-webui/assets/71559822/ced032dd-3dda-4440-b72e-3f281d146e56)

**Note**

This library is modified from [url](https://github.com/mkshing/notebooks/blob/main/stable_video_diffusion_img2vid.ipynb), which is the colab version of stable-video-diffusion.