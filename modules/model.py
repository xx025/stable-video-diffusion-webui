import math
import os
import random
from glob import glob
from pathlib import Path
from typing import Optional, Any

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from sgm.inference.helpers import embed_watermark
from torch import Tensor
from torchvision.transforms import ToTensor

from modules.img_tool import create_video, crop_center_resize
from modules.img_tool import enlarge_image, shrink_image, crop_to_nearest_multiple_of_n, \
    image_pipeline_func
from modules.model_setting import device, model, filter_x
from setting import creat_video_by_opencv
from setting import img_resize_to_HW, auto_adjust_img
from setting import vid_output_folder


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device, dtype=None):
    batch = {}
    batch_uc: dict[str | Any, Tensor] = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device, dtype=dtype)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device, dtype=dtype)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device, dtype=dtype),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def sample(
        input_path: str = "assets/test_image.png",  # Can either be image file or folder with image files
        resize_image: bool = False,
        num_frames: Optional[int] = None,
        num_steps: Optional[int] = None,
        fps_id: int = 6,
        motion_bucket_id: int = 127,
        cond_aug: float = 0.02,
        seed: int = 23,
        decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
        device: str = "cuda",
        output_folder: Optional[str] = "/content/outputs",
        skip_filter: bool = False
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    torch.manual_seed(seed)
    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError
    all_out_paths = []
    for input_img_path in all_img_paths:
        # 图片数据管道，在这里会进行一些图片的预处理，比如缩放，剪裁等
        # Image data pipeline, some image preprocessing will be done here, such as scaling, cropping, etc.
        min_W, min_H = auto_adjust_img.get('min_width', 256), auto_adjust_img.get('min_height', 256)
        max_W, max_H = auto_adjust_img.get('max_width', 1024), auto_adjust_img.get('max_height', 1024)
        re_W, re_H = img_resize_to_HW.get(' target_width', 1024), img_resize_to_HW.get('target_height', 576)
        multiple_of_N = img_resize_to_HW.get('multiple_of_N', 64)
        processing_functions = [
            {"func": enlarge_image, "args": (min_H, min_W)},
            {"func": shrink_image, "args": (max_W, max_H)},
            {"func": crop_to_nearest_multiple_of_n, "args": (multiple_of_N,)},
        ]
        image = image_pipeline_func(input_img_path, processing_functions)
        if resize_image and image.size != (re_W, re_H):
            image = crop_center_resize(image, target_width=re_W, target_height=re_H)

        if image.mode == "RGBA":
            image = image.convert("RGB")

        image = ToTensor()(image)
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)

        if (H, W) != (576, 1024):
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)

        # low vram mode
        model.conditioner.cpu()
        model.first_stage_model.cpu()
        torch.cuda.empty_cache()
        model.sampler.verbose = True

        with torch.no_grad():
            with torch.autocast(device):
                model.conditioner.to(device)
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )
                model.conditioner.cpu()
                torch.cuda.empty_cache()

                # from here, dtype is fp16
                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)
                for k in uc.keys():
                    uc[k] = uc[k].to(dtype=torch.float16)
                    c[k] = c[k].to(dtype=torch.float16)

                randn = torch.randn(shape, device=device, dtype=torch.float16)
                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(2, num_frames).to(device, )
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                for k in additional_model_inputs:
                    if isinstance(additional_model_inputs[k], torch.Tensor):
                        additional_model_inputs[k] = additional_model_inputs[k].to(dtype=torch.float16)

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                samples_z.to(dtype=model.first_stage_model.dtype)

                model.en_and_decode_n_samples_a_time = decoding_t
                model.first_stage_model.to(device)
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                model.first_stage_model.cpu()
                torch.cuda.empty_cache()

                os.makedirs(output_folder, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder, "*.mp4")))
                video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")

                samples = embed_watermark(samples)
                if not skip_filter:
                    samples = filter_x(samples)
                else:
                    print("WARNING: You have disabled the NSFW/Watermark filter. "
                          "Please do not expose unfiltered results in services or applications open to the public.")

                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                if creat_video_by_opencv:
                    writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps_id + 1,  #
                        (samples.shape[-1], samples.shape[-2]),
                    )
                    for frame in vid:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writer.write(frame)
                    writer.release()
                else:
                    create_video(vid, video_path, fps_id)
                all_out_paths.append(video_path)
    return all_out_paths


def infer(
        input_path: str,
        resize_image: bool,
        n_frames: int,
        n_steps: int,
        seed: str,
        decoding_t: int,
        fps_id: int,
        motion_bucket_id: int,
        cond_aug: float,
        skip_filter: bool = False) -> str:
    if seed == "random":
        seed = random.randint(0, 2 ** 32)
    seed = int(seed)
    output_folder = vid_output_folder
    output_paths = sample(
        input_path=input_path,
        resize_image=resize_image,
        num_frames=n_frames,
        num_steps=n_steps,
        fps_id=fps_id,
        motion_bucket_id=motion_bucket_id,
        cond_aug=cond_aug,
        seed=seed,
        decoding_t=decoding_t,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
        device=device,
        output_folder=output_folder,
        skip_filter=skip_filter,
    )
    return output_paths[0]
