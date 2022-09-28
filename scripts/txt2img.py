import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from transformers import CLIPTokenizer, CLIPTextModel

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# exiftool.exeにパスを通すまたは同ディレクトリに配置
# https://exiftool.org
# pyexiftoolをインストールする
# https://pypi.org/project/PyExifTool/
from exiftool import ExifTool
from exiftool import ExifToolHelper
import random
import csv

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: © 2022 lox9973
# https://gitlab.com/-/snippets/2395088
def patch_conv(klass):
    init = klass.__init__

    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, padding_mode="circular")

    klass.__init__ = __init__


# waifu-diffusionは他モデルとプロンプトの順番が違うらしい
# https://wiki.installgentoo.com/wiki/Stable_Diffusion#Waifu_Diffusion
def load_prompt_csv(path, deli):
    with open(path, encoding="utf_8") as f:
        reader = csv.reader(f)
        raw_list = [row for row in reader]
        mat = np.array(raw_list[1:]).transpose().tolist()
        prompts = []
        for vec in mat:
            prompts.append(deli.join(sorted([i for i in vec if i != ""])))
        prompts = deli.join([i for i in prompts if i != ""])
    return prompts


# seed値再現でプロンプト取得
def read_metadata(path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata(path)
        n_prompt = ""
        seed = metadata[0]["XMP:Title"]
        prompt = metadata[0]["XMP:Creator"]
        if "XMP:Subject" in metadata[0]:
            n_prompt = metadata[0]["XMP:Subject"]
        return seed, prompt, n_prompt


# token長さチェック
def decode_token_matrix(tokenizer, token):
    if isinstance(token, (list, tuple)):
        decoded = []
        for t in token:
            decoded.append(decode_token_matrix(tokenizer, t))
        return decoded
    else:
        return tokenizer.decoder.get(token)


def check_token_length(tokenizer, token):
    max_length = 77
    end_of_text = 49407

    batch_encoding = tokenizer(
        token,
        truncation=True,
        max_length=max_length,
        return_length=True,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt",
    )
    accept_tokens = batch_encoding.input_ids.tolist()[0]
    end_pos = accept_tokens.index(end_of_text) + 1
    print(f"accept_token = {end_pos}")
    print(decode_token_matrix(tokenizer, accept_tokens[:end_pos]))
    if hasattr(batch_encoding, "overflowing_tokens"):
        overflowing_tokens = batch_encoding.overflowing_tokens
        print(f"overflowing_tokens = {overflowing_tokens.size()[-1]}")
        print(decode_token_matrix(tokenizer, overflowing_tokens.tolist()[0]))


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def main():
    parser = argparse.ArgumentParser()
    default_prompt = "a painting of a virus monster playing guitar"

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default=default_prompt,
        help="the prompt to render",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action="store_true",
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action="store_true",
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action="store_true",
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )
    # メモリ節約
    parser.add_argument("--fp16", action="store_true", help="convert model to fp16")
    # タイリングテクスチャ
    parser.add_argument("--tile", action="store_true", help="create tiling texture")
    # プロンプトの加減算
    # https://zenn.dev/td2sk/articles/eb772103a3a8ff
    parser.add_argument(
        "--prompt-correction",
        action="append",
        help="prompt correction: 'word::0.2' 'word::-0.1' 'word, other word::0.1'",
    )
    # seedをまとめて再現
    parser.add_argument(
        "--rep_seed",
        action="append",
        help="reproduce the seeds",
    )
    # seedをまとめて再現2
    parser.add_argument(
        "--rep_dir",
        type=str,
        help="reproduce the seeds from dir",
    )
    # promptをcsvファイルから読み込む
    parser.add_argument(
        "--prompt_csv",
        type=str,
        help="load prompts from this csv. this feature is for waifu-diffusion",
    )
    # Negative prompt
    parser.add_argument(
        "--negative_prompt",
        type=str,
        help="negative prompts",
    )
    # Negative prompt csv
    parser.add_argument(
        "--negative_prompt_csv",
        type=str,
        help="negative prompts",
    )
    parser.add_argument(
        "--check_token_length", action="store_true", help="check token length"
    )

    opt = parser.parse_args()

    if opt.tile:
        for klass in [torch.nn.Conv2d, torch.nn.ConvTranspose2d]:
            patch_conv(klass)

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        if not opt.prompt_csv:
            prompt = opt.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]
        else:
            print(f"\nreading prompts from {opt.prompt_csv}")
            prompt = load_prompt_csv(opt.prompt_csv, " ")
            assert prompt is not None
            data = [batch_size * [prompt]]
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    if opt.negative_prompt_csv:
        print(f"reading negative_prompts from {opt.negative_prompt_csv}")
        n_prompt = load_prompt_csv(opt.negative_prompt_csv, ", ")
        assert n_prompt is not None
        n_data = [batch_size * [n_prompt]]
    elif opt.negative_prompt:
        n_prompt = opt.negative_prompt
        assert n_prompt is not None
        n_data = [batch_size * [n_prompt]]
    else:
        n_data = [batch_size * [""]]

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    print("prompts")
    check_token_length(tokenizer, data[0])
    print("negative prompts")
    check_token_length(tokenizer, n_data[0])

    if opt.check_token_length:
        return

    seed_everything(opt.seed)
    seed = opt.seed

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # メモリ節約
    if opt.fp16:
        model = model.half()
    else:
        model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    # 引数保存
    txt_path = os.path.join(outpath, "txt")
    os.makedirs(txt_path, exist_ok=True)

    base_count = len(os.listdir(sample_path))
    base_begin = base_count
    grid_count = len(os.listdir(outpath)) - 1

    rep_files = []
    if opt.rep_dir:
        rep_files = glob.glob(opt.rep_dir + "/*.png", recursive=False)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn(
            [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device
        )

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    random.seed()

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()

                n_iter = opt.n_iter
                if opt.rep_seed:
                    n_iter = len(opt.rep_seed)
                elif opt.rep_dir:
                    n_iter = len(rep_files)
                i = 0
                for n in trange(n_iter, desc="Sampling"):
                    if opt.rep_seed or opt.rep_dir or n_iter > 1:
                        if opt.rep_seed:
                            seed = opt.rep_seed[i]
                        elif opt.rep_dir:
                            seed, prompt, n_prompt = read_metadata(rep_files[i])
                            if not (opt.prompt != default_prompt or opt.prompt_csv):
                                assert prompt is not None
                                data = [batch_size * [prompt]]
                            if not (opt.negative_prompt or opt.negative_prompt_csv):
                                n_data = [batch_size * [n_prompt]]

                        else:
                            seed = random.randint(0, 0x7FFFFFFF)
                        i += 1
                        print("\n")
                        seed_everything(seed)

                    for prompts in tqdm(data, desc="data"):
                        uc = None

                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if (
                            opt.negative_prompt_csv
                            or opt.negative_prompt
                            or opt.rep_dir
                        ):
                            n_prompts = n_data[0]
                            if isinstance(n_prompts, tuple):
                                n_prompts = list(n_prompts)
                            uc = model.get_learned_conditioning(n_prompts)

                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        # プロンプトの加減算
                        correction_weight = 1
                        if opt.prompt_correction:
                            for pw in opt.prompt_correction:
                                for pw in opt.prompt_correction:
                                    pw = pw.split("::")
                                    p, weight = pw[:-1], float(pw[-1])
                                    correction_weight += weight
                                    c += weight * model.get_learned_conditioning(
                                        list(p)
                                    )
                        c = c / correction_weight

                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(
                            S=opt.ddim_steps,
                            conditioning=c,
                            batch_size=opt.n_samples,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=opt.scale,
                            unconditional_conditioning=uc,
                            eta=opt.ddim_eta,
                            x_T=start_code,
                        )

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                        )
                        x_samples_ddim = (
                            x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        )

                        x_checked_image = x_samples_ddim

                        x_checked_image_torch = torch.from_numpy(
                            x_checked_image
                        ).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255.0 * rearrange(
                                    x_sample.cpu().numpy(), "c h w -> h w c"
                                )
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                sample_base_count = os.path.join(
                                    sample_path, f"{base_count:05}.png"
                                )
                                img.save(sample_base_count)
                                base_count += 1
                                # 引数保存
                                with ExifTool() as et:
                                    print(
                                        et.execute(
                                            *[
                                                "-XMP-dc:description=seed={seed}, param={opt}"
                                            ]
                                            + ["-XMP-dc:title={seed}"]
                                            + ["-XMP-dc:creator={prompts}"]
                                            + ["-XMP-dc:subject={n_prompts}"]
                                            + ["-overwrite_original"]
                                            + [sample_base_count]
                                        )
                                    )
                                print("imave save -> ", str(sample_base_count))

                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, "n b c h w -> (n b) c h w")
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    grid_name = os.path.join(outpath, f"grid-{grid_count:05}.png")
                    img.save(grid_name)
                    # 引数保存
                    base_range = f"{base_begin:05}.png:{base_count-1:05}.png"
                    with ExifTool() as et:
                        print(
                            et.execute(
                                *["-XMP-dc:description={base_range}, param={opt}"]
                                + ["-overwrite_original"]
                                + [grid_name]
                            )
                        )
                    with open(
                        os.path.join(txt_path, f"grid-{grid_count:04}.txt"), mode="w"
                    ) as f:
                        f.write("{opt}\n")
                    print("imave save -> {grid_name}")
                    print("  base_range : {base_range}")

                    grid_count += 1

                toc = time.time()

    print(
        f"Your samples are ready and waiting for you here: \n{outpath} \n" f" \nEnjoy."
    )


if __name__ == "__main__":
    main()
