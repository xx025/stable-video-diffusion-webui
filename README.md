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
python run.py
```

## More setting

Please check settings.py and make modifications.

---

![捕获](https://github.com/xx025/stable-video-diffusion-webui/assets/71559822/01750957-a9b4-4d28-9938-578336c2fa90)


## Communicate Group

> Only for China users

<img src="https://github.com/xx025/stable-video-diffusion-webui/assets/71559822/2608e9c2-c641-4932-9244-90eec2e9d8f5" width = "400" />

**Note**

This library is modified from [url](https://github.com/mkshing/notebooks/blob/main/stable_video_diffusion_img2vid.ipynb), which is the colab version of stable-video-diffusion.
