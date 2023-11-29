import subprocess

import sys


def install_dependencies():
    try:
        # 克隆 generative-models 仓库
        subprocess.run(['git', 'clone', 'https://github.com/Stability-AI/generative-models.git'])
        # 从 requirements.txt 安装依赖项
        subprocess.run(['pip3', 'install', '-r', 'requirements.txt'])
        subprocess.run(['pip3', 'install', '-e', 'generative-models'])  # 安装 generative-models
        # 安装 sdata，指定 git 分支
        subprocess.run(
            ['pip3', 'install', '-e', 'git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata'])
        # 安装 gradio
        subprocess.run(['pip3', 'install', 'gradio'])
        platform = sys.platform  # 获取操作系统类型
        if platform == "win32":  # Windows-specific packages
            subprocess.run(['pip', 'install',
                            'https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl'])
        elif platform == "linux":  # Linux-specific packages
            subprocess.run(['pip', 'install', 'triton==2.0.0'])
        else:
            print(f"Unsupported operating system: {platform}")

        print("Dependencies installed successfully.")
    except Exception as e:
        print(f"Error installing dependencies: {e}")


if __name__ == "__main__":
    install_dependencies()
