# MiniMax H3 Semantic Bridge for ComfyUI

A compact conditioning-space adapter for the standard MiniMax H3 FL2VA and
text-conditioned generation path. It was distilled from a cross-architecture
semantic mapping developed with SenseNova U1.5 as a research teacher.
SenseNova is not required at inference time.

This is an independent experimental project and is not an official MiniMax or
SenseNova release.

## Scope

- Supported: MiniMax H3 FL2VA, text-to-video, first-frame and last-frame paths.
- Not supported in v1: Ref2VA or other reference-conditioned workflows.
- Do not use v1 for reference-audio singing or lip-sync. Experimental tests
  showed worse vocal articulation when the bridge was applied to Ref2VA
  conditioning.

## Installation

### ComfyUI Manager

Search for **MiniMax H3 Semantic Bridge** in ComfyUI Manager, install it, and
restart ComfyUI.

### Manual installation

Clone or extract this repository into:

```text
ComfyUI/custom_nodes/MiniMax-H3-Semantic-Bridge/
```

Install the small Python dependency if your ComfyUI environment does not
already provide it:

```bash
pip install -r requirements.txt
```

Restart ComfyUI after installation.

## Adapter download

The adapter is distributed separately on Hugging Face:

https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge

Download:

```text
MiniMaxH3_SemanticBridge_v1.safetensors
```

Place it at:

```text
ComfyUI/models/semantic_bridge/MiniMaxH3_SemanticBridge_v1.safetensors
```

Then restart ComfyUI or refresh its model lists.

## Included nodes

- **MiniMax H3 Image to Video + Semantic Bridge**
- **MiniMax H3 Semantic Bridge**
- **MiniMax H3 Clear Semantic Bridge Cache**

The nodes appear under:

```text
MiniMax H3/Semantic Bridge
```

## Recommended settings

| Setting | Recommended value | Notes |
| --- | --- | --- |
| `alpha` | `0.10` | General starting point |
| `magnitude_match` | `per_token` | Recommended mode |
| `alpha` | `0.15` | Stronger influence used in published A/B examples |

The adapter transforms native H3 conditioning and blends the predicted
semantic representation back into it. It does not merge weights into the H3
diffusion model.

## Research, examples and limitations

The complete research narrative, metrics, datasets, scripts, controlled A/B
examples, workflow, and detailed limitations are published on the
[Hugging Face project page](https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge).

This is an experimental adapter, not a universal enhancer. It may help some
prompts, make little difference on others, or occasionally make a result worse.
Representation-space similarity metrics are not equivalent to perceptual video
quality.

## License

Review [LICENSE.md](LICENSE.md), [NOTICE.txt](NOTICE.txt), and
[UPSTREAM_LICENSES.md](UPSTREAM_LICENSES.md) before redistribution or commercial
use. MiniMax H3 and model-derived artifacts are subject to the upstream MiniMax
H3 Community License Agreement.
