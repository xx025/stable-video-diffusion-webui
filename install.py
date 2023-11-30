import os
import shutil
import subprocess
import sys


def install_dependencies():
    try:
        # 克隆 generative-models 仓库
        repo_url = 'https://github.com/Stability-AI/generative-models.git'
        commit_hash = 'e6f0e36f5e856d9651c597d75aed13ae7298d03b'
        # 使用 subprocess.run 克隆存储库并切换到指定提交
        subprocess.run(['git', 'clone', repo_url])
        subprocess.run(['git', 'checkout', commit_hash], cwd='generative-models')  # 切换到指定提交
        print(f"Repository cloned and checked out to commit {commit_hash}")

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
            subprocess.run([
                'pip',
                'install',
                'https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl'])
        elif platform == "linux":  # Linux-specific packages
            subprocess.run(['pip', 'install', 'triton==2.0.0'])
        else:
            print(f"Unsupported operating system: {platform}")

        # 复制必要的文件
        RESOURCES_ROOT = "scripts/util/detection/"
        os.makedirs(RESOURCES_ROOT, exist_ok=True)
        f_real = os.path.join('generative-models', RESOURCES_ROOT)
        files = ['p_head_v1.npz', 'w_head_v1.npz']
        for f in files:
            if not os.path.exists(os.path.join(RESOURCES_ROOT, f)):
                shutil.copy(os.path.join(f_real, f), os.path.join(RESOURCES_ROOT, f))

        # 创建
        # os.makedirs('checkpoints', exist_ok=True)
        print("Dependencies installed successfully.")
    except Exception as e:
        print(f"Error installing dependencies: {e}")


if __name__ == "__main__":
    install_dependencies()
