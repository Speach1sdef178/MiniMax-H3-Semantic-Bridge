
import os
import gc

import torch
import torch.nn as nn
from safetensors.torch import load_file

import folder_paths
import node_helpers
import comfy_extras.nodes_minimax_h3 as native_h3


DISTILLED_DIR = os.path.join(folder_paths.models_dir, "semantic_bridge")
os.makedirs(DISTILLED_DIR, exist_ok=True)

try:
    folder_paths.add_model_folder_path("semantic_bridge", DISTILLED_DIR)
except Exception:
    pass

_STUDENT_CACHE = {}


def _list_safetensors():
    try:
        files = folder_paths.get_filename_list("semantic_bridge")
    except Exception:
        files = []
    files = [x for x in files if x.lower().endswith(".safetensors")]
    return sorted(files) if files else ["NO_DISTILLED_ADAPTER_FOUND.safetensors"]


def _full_adapter_path(name):
    path = None
    try:
        path = folder_paths.get_full_path("semantic_bridge", name)
    except Exception:
        pass
    if path and os.path.isfile(path):
        return path
    fallback = os.path.join(DISTILLED_DIR, name)
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(
        "Distilled adapter was not found.\n\n"
        f"Selected: {name}\nExpected folder:\n{DISTILLED_DIR}\n"
    )


def _rms_normalize(x):
    x = x.float()
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return x / rms


def _magnitude_match(source, target):
    source = source.float()
    target = target.float()
    source_rms = torch.sqrt(source.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
    target_rms = torch.sqrt(target.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
    return source * (target_rms / source_rms)


class SemanticStudent(nn.Module):
    def __init__(self, input_dim=5120, hidden_dim=512, output_dim=5120):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=True)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


def _load_student(adapter_name, device):
    path = _full_adapter_path(adapter_name)
    key = (os.path.abspath(path), str(device))
    cached = _STUDENT_CACHE.get(key)
    if cached is not None:
        return cached

    weights = load_file(path, device="cpu")
    required = [
        "fc1.weight", "fc1.bias",
        "fc2.weight", "fc2.bias",
        "fc3.weight", "fc3.bias",
    ]
    missing = [k for k in required if k not in weights]
    if missing:
        raise RuntimeError(f"Invalid distilled adapter. Missing tensors: {missing}")

    if tuple(weights["fc1.weight"].shape) != (512, 5120):
        raise RuntimeError(f"Unexpected fc1.weight shape: {tuple(weights['fc1.weight'].shape)}")
    if tuple(weights["fc2.weight"].shape) != (512, 512):
        raise RuntimeError(f"Unexpected fc2.weight shape: {tuple(weights['fc2.weight'].shape)}")
    if tuple(weights["fc3.weight"].shape) != (5120, 512):
        raise RuntimeError(f"Unexpected fc3.weight shape: {tuple(weights['fc3.weight'].shape)}")

    model = SemanticStudent()
    with torch.no_grad():
        model.fc1.weight.copy_(weights["fc1.weight"].float())
        model.fc1.bias.copy_(weights["fc1.bias"].float())
        model.fc2.weight.copy_(weights["fc2.weight"].float())
        model.fc2.bias.copy_(weights["fc2.bias"].float())
        model.fc3.weight.copy_(weights["fc3.weight"].float())
        model.fc3.bias.copy_(weights["fc3.bias"].float())

    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    _STUDENT_CACHE[key] = model
    print(f"[MiniMax H3 Semantic Bridge] Loaded adapter: {adapter_name}")
    return model


def _apply_distilled_bridge(conditioning, adapter_name, alpha, magnitude_match):
    result = []

    for item in conditioning:
        if len(item) != 2:
            raise RuntimeError("Unexpected ComfyUI CONDITIONING structure.")

        native = item[0]
        metadata = item[1]

        if native.ndim != 3 or native.shape[-1] != 5120:
            raise RuntimeError(
                "Expected MiniMax H3 conditioning [B,T,5120], "
                f"got {tuple(native.shape)}"
            )

        student = _load_student(adapter_name, native.device)
        h = native.float()
        x = _rms_normalize(h)

        with torch.inference_mode():
            projected = student(x)

        if magnitude_match == "per_token":
            projected = _magnitude_match(projected, h)
        elif magnitude_match == "global":
            source_rms = torch.sqrt(projected.pow(2).mean() + 1e-8)
            target_rms = torch.sqrt(h.pow(2).mean() + 1e-8)
            projected = projected * (target_rms / source_rms)

        hybrid = h + float(alpha) * (projected - h)
        hybrid = hybrid.to(dtype=native.dtype)

        new_metadata = dict(metadata)
        new_metadata["sensenova_h3_distilled"] = True
        new_metadata["sensenova_h3_distilled_alpha"] = float(alpha)
        new_metadata["sensenova_h3_distilled_mode"] = magnitude_match
        new_metadata["sensenova_h3_distilled_adapter"] = adapter_name

        result.append([hybrid, new_metadata])

    return result


class MiniMaxH3DistilledImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": "",
                }),
                "distilled_adapter": (_list_safetensors(),),
                "alpha": ("FLOAT", {
                    "default": 0.10,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "magnitude_match": (
                    ["per_token", "global", "none"],
                    {"default": "per_token"},
                ),
                "width": ("INT", {
                    "default": 1344,
                    "min": 32,
                    "max": 16384,
                    "step": 32,
                }),
                "height": ("INT", {
                    "default": 768,
                    "min": 32,
                    "max": 16384,
                    "step": 32,
                }),
                "length": ("INT", {
                    "default": 124,
                    "min": 5,
                    "max": 3600,
                    "step": 17,
                }),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "generate"
    CATEGORY = "MiniMax H3/Semantic Bridge"

    def generate(
        self,
        clip,
        vae,
        prompt,
        distilled_adapter,
        alpha,
        magnitude_match,
        width,
        height,
        length,
        first_frame=None,
        last_frame=None,
    ):
        if not prompt.strip():
            raise ValueError("Prompt is empty.")

        latent, frame_count = native_h3._empty_av_latent(width, height, length)

        images = []
        keyframes = []

        if first_frame is not None:
            img = native_h3._resize(first_frame[:1], width, height, "disabled")
            images.append(img)
            keyframes.append({
                "resolved_frame_index": 0,
                "image": img,
            })

        if last_frame is not None:
            img = native_h3._resize(last_frame[:1], width, height, "center")
            images.append(img)
            keyframes.append({
                "resolved_frame_index": frame_count - 1,
                "image": img,
            })

        tokens = clip.tokenize(prompt, images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)

        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))

            cond = node_helpers.conditioning_set_values(
                cond,
                {
                    "minimax_keyframes": keyframes,
                    "minimax_frame_count": frame_count,
                },
            )

        positive = _apply_distilled_bridge(
            cond,
            distilled_adapter,
            alpha,
            magnitude_match,
        )

        return (positive, latent)


class SenseNovaH3DistilledBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "distilled_adapter": (_list_safetensors(),),
                "alpha": ("FLOAT", {
                    "default": 0.10,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "magnitude_match": (
                    ["per_token", "global", "none"],
                    {"default": "per_token"},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/Semantic Bridge"

    def apply(self, conditioning, distilled_adapter, alpha, magnitude_match):
        return (
            _apply_distilled_bridge(
                conditioning,
                distilled_adapter,
                alpha,
                magnitude_match,
            ),
        )


class SenseNovaH3ClearDistilledCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clear": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Semantic Bridge"

    def run(self, clear):
        if clear:
            _STUDENT_CACHE.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return ("MiniMax H3 Semantic Bridge adapter cache cleared.",)
        return ("Cache unchanged.",)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3DistilledImageToVideo": MiniMaxH3DistilledImageToVideo,
    "SenseNovaH3DistilledBridge": SenseNovaH3DistilledBridge,
    "SenseNovaH3ClearDistilledCache": SenseNovaH3ClearDistilledCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3DistilledImageToVideo":
        "MiniMax H3 Image to Video + Semantic Bridge",
    "SenseNovaH3DistilledBridge":
        "MiniMax H3 Semantic Bridge",
    "SenseNovaH3ClearDistilledCache":
        "MiniMax H3 Clear Semantic Bridge Cache",
}
