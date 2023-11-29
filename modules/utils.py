import torch
from omegaconf import OmegaConf
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.util import instantiate_from_config


def load_model(config: str, device: str, num_frames: int, num_steps: int, ):
    config = OmegaConf.load(config)
    config.model.params.conditioner_config.params.emb_models[
        0
    ].params.open_clip_embedding_config.params.init_device = device
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    with torch.device(device):
        model = instantiate_from_config(config.model).to(device).eval().requires_grad_(False)
    filter_1 = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter_1
