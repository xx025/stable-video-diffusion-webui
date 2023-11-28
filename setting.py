import os

# 首次运行需要从Hugging Face下载ClIP模型等 也需要设置代理
# 设置自己的 http代理
# This config about proxy only for China
# os.environ['http_proxy'] = 'http://127.0.0.1:2233/'
# os.environ['https_proxy'] = 'http://127.0.0.1:2233/'

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定显卡

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
