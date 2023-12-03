import os

# 首次运行需要从Hugging Face下载ClIP模型等 也需要设置代理
# 设置自己的 http代理
# This config about proxy only for China
# os.environ['http_proxy'] = 'http://127.0.0.1:7890/'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890/'

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 指定英伟达显卡
# Specify the NVIDIA GPU to use


gradio_args = dict(
    share=True,
    # 产生一个gradio 的分享链接，可以通过gradio的网站进行访问
    # generate a gradio share link, you can access it through the gradio website
    auth=dict(
        # username='root',
        # password='123456',
        message='Place enter your username and password; 请输入用户名何密码'
    ),
    # 设置用户名何密码， 如果你不想设置请注释掉
    # set username and password， if you don't want to set it, please comment it out
    head_html="""
            <div style="text-align: center;line-height:0">
                <h1>Stable Video Diffusion WebUI</h1>
                <p>Upload an image to create a Video with the image.</p>
            </div>
            """,
    # 设置gradio的头部信息
    # Set the header information of gradio
    show_api=True,
    # 显示api 信息 show api information
)

auto_adjust_img = dict(
    min_width=256,  # 图片最小宽度 Image minimum width
    min_height=256,  # 图片最小高度 Image minimum height
    max_height=1024,  # 图片最大宽度 Image maximum width
    max_width=1024,  # 图片最大高度 Image maximum height
    multiple_of_N=64  # 图片的宽高必须是N的倍数 The width and height of the image must be a multiple of N
)
# 自动调整图片分辨率,自动调整到符合要求的分辨率
# Automatically adjust the image resolution, automatically adjust to the resolution that meets the requirements


img_resize_to_HW = dict(
    target_width=1024,  # 目标宽度
    target_height=576,  # 目标高度
)
# 因为往往在训练尺寸下的图片尺寸能达到比较好的效果,但是硬剪裁会扭曲图片，所以使用从中心剪裁
# Because often, using images at training size can achieve better
# results, but hard cropping can distort the images, so we use center cropping.


creat_video_by_opencv = False
# 使用opencv生成视频, 但是发现会有一些编码的问题，所以默认关闭，默认使用moviepy
# Use opencv to generate video, but it is found that there will be some encoding problems,
# so it is turned off by default,default use moviepy


vid_output_folder = 'content/outputs'
# 生成视频的输出文件夹
# Output Folder for Generated Videos


infer_args = dict(
    default_fps=6,
    # 默认的视频帧率
    # Default video fps
    nsfw_filter_checkbox=dict(
        enable=True,
        default=True
    )
    # 是否可以跳过nsfw过滤, 默认打开此选择,
    # 简而言之小心别人用你部署的服务推理不良内容，建议enable=False 关掉这个选项 并且设置default=False
    # Whether nsfw filtering can be skipped
)
# 推理参数
# Inference parameters
