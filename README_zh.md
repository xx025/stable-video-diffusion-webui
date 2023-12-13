# Stable-Video-Diffusion-WebUI


[English](README.md) | [中文](README_zh.md)



|![284758167-367a25d8-8d7b-42d3-8391-6d82813c7b0f](https://github.com/xx025/stable-video-diffusion-webui/assets/71559822/00a0be47-a6e6-4c77-9bbc-ff8b5d899cfe) |![000003](https://github.com/xx025/stable-video-diffusion-webui/assets/71559822/8f53ac31-04fe-4e78-bbe0-625e690267cc) |
|---------------------------------|----------------------------------|
| img                     | videos                      |



## AutoDl平台Docker 使用

这样你不需要安装了，[Docker](https://www.codewithgpu.com/i/xx025/stable-video-diffusion-webui/stable-video-diffusion-webui)



## 准备



```shelll
git clone https://github.com/xx025/stable-video-diffusion-webui.git
cd stable-video-diffusion-webui
```
再正式开始之前你需要创建一个虚拟环境，推荐使用conda
如果你不方便使用 conda 请用自己的方式创建一个 python3.10 版本的环境

```shell
conda create -n svd python=3.10 
conda activate svd
```

然后依次执行以下命令，
> 很有可能在这个过程中你需要流畅的访问国际网络的上网环境,请合理配置代理

> 一种代理设置方法：[使用xshell 隧道让服务器通过本地代理上网](https://github.com/xx025/stable-video-diffusion-webui/wiki/%E4%BD%BF%E7%94%A8xshell-%E9%9A%A7%E9%81%93%E8%AE%A9%E6%9C%8D%E5%8A%A1%E5%99%A8%E9%80%9A%E8%BF%87%E6%9C%AC%E5%9C%B0%E4%BB%A3%E7%90%86%E4%B8%8A%E7%BD%91)

```shell

# 在此过程中会安装 cu118对应的torch版本，你可以先尝试安装，笔者测试机器cuda11.6 可正常安装 
# 如果不适你的合你的电脑请合理更改requirements.txt中的--extra-index-url
python install.py

# 创建文件夹，保存权重文件
mkdir checkpoints 
# 下载模型，这个可能需要一些时间，
# 如果不方便使用wget 可以手动下载保存到 checkpoints 文件名 命名为 svd_xt.safetensors
wget https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors?download=true -P checkpoints/ -O svd_xt.safetensors
```

## 运行

在首次运行时，会下载其他的模型文件，

> 请为合理设置系统设置代理或修改proxy.py 的代理配置

```shell
python run.py
```

## 更多设置

请查看`setting.py` 并修改


---

![捕获](https://github.com/xx025/stable-video-diffusion-webui/assets/71559822/01750957-a9b4-4d28-9938-578336c2fa90)


## 交流社群

<img src="https://github.com/xx025/stable-video-diffusion-webui/assets/71559822/2608e9c2-c641-4932-9244-90eec2e9d8f5" width = "400" />


**说明**

这个库来修改自 [url](https://github.com/mkshing/notebooks/blob/main/stable_video_diffusion_img2vid.ipynb)  , 它是 stable-video-diffusion的 colab 版本



