import torch

from modules.utils import load_model

version = "svd_xt"  # @param ["modules", "svd_xt"]

if version == "modules":
    num_frames = 14
    num_steps = 25
    # output_folder = default(output_folder, "outputs/simple_video_sample/modules/")
    model_config = "generative-models/scripts/sampling/configs/modules.yaml"
elif version == "svd_xt":
    num_frames = 25
    num_steps = 30
    # output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
    model_config = "generative-models/scripts/sampling/configs/svd_xt.yaml"
else:
    raise ValueError(f"Version {version} does not exist.")

device = "cuda" if torch.cuda.is_available() else "cpu"

model, filter_x = load_model(
    model_config,
    device,
    num_frames,
    num_steps,
)
