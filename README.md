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
conda create -n modules python=3.10 
conda activate modules
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
python img2vid.py
```

## More setting

Please check settings.py and make modifications.


![image](https://github.com/xx025/stable-video-diffusion-webui/assets/71559822/ced032dd-3dda-4440-b72e-3f281d146e56)

**Note**

This library is modified from [url](https://github.com/mkshing/notebooks/blob/main/stable_video_diffusion_img2vid.ipynb), which is the colab version of stable-video-diffusion.