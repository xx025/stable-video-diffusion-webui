import os

# 首次运行需要从Hugging Face下载ClIP模型等 也需要设置代理
# 设置自己的 http代理
# This config about proxy only for China
# os.environ['http_proxy'] = 'http://127.0.0.1:2233/'
# os.environ['https_proxy'] = 'http://127.0.0.1:2233/'

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 指定英伟达显卡
# Specify the NVIDIA GPU to use

GR_SHARE = True
# 产生一个gradio 的分享链接，可以通过gradio的网站进行访问
# generate a gradio share link, you can access it through the gradio website

auth = dict(
    # USER_NAME='root',
    # PASSWORD='123456',
    AUTH_MESSAGE='Place enter your username and password; 请输入用户名何密码',
)
# 设置用户名何密码， 如果你不想设置请注释掉
# set username and password， if you don't want to set it, please comment it out

auto_adjust_img = dict(
    enable=True,
    max_height=1024,  # 图片最大宽度
    max_width=1024,  # 图片最大高度
)
# 自动调整图片分辨率
# Automatically adjust the resolution of the picture


img_crop_center = dict(
    enable=True,
    target_width=1024,  # 目标宽度
    target_height=576,  # 目标高度
)
# 因为往往在训练尺寸下的图片尺寸能达到比较好的效果,但是硬剪裁会扭曲图片，所以使用从中心剪裁 Because often, using images at training size can achieve better
# results, but hard cropping can distort the images, so we use center cropping.


creat_video_by_opencv = False
# 使用opencv生成视频, 但是发现会有一些编码的问题，所以默认关闭，默认使用moviepy
# Use opencv to generate video, but it is found that there will be some encoding problems,
# so it is turned off by default,default use moviepy
