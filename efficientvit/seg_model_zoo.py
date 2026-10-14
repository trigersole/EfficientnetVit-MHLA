from functools import partial
from typing import Callable, Optional

from efficientvit.models.efficientvit import (
    EfficientViTSeg,
    efficientvit_seg_b0,
    efficientvit_seg_b1,
    efficientvit_seg_b2,
    efficientvit_seg_b3,
    efficientvit_seg_l1,
    efficientvit_seg_l2,
)
from efficientvit.models.nn.norm import set_norm_eps
from efficientvit.models.utils import load_state_dict_from_file

__all__ = [
    "EFFICIENTVIT_SEG_EXPERIMENTS",
    "create_efficientvit_seg_experiment_model",
    "create_efficientvit_seg_model",
]


EFFICIENTVIT_SEG_EXPERIMENTS = {
    "e0": {"attention_type": "lite_mla", "use_multiscale_qkv": True},
    "e1": {"attention_type": "lite_mla", "use_multiscale_qkv": False},
    "e2": {"attention_type": "mhla", "use_multiscale_qkv": False},
    "e3": {"attention_type": "mhla", "use_multiscale_qkv": True},
}


REGISTERED_EFFICIENTVIT_SEG_MODEL: dict[str, tuple[Callable, float, str]] = {
    "efficientvit-seg-b0-cityscapes": (
        partial(efficientvit_seg_b0, dataset="cityscapes"),
        1e-5,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_b0_cityscapes.pt",
    ),
    "efficientvit-seg-b1-cityscapes": (
        partial(efficientvit_seg_b1, dataset="cityscapes"),
        1e-5,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_b1_cityscapes.pt",
    ),
    "efficientvit-seg-b2-cityscapes": (
        partial(efficientvit_seg_b2, dataset="cityscapes"),
        1e-5,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_b2_cityscapes.pt",
    ),
    "efficientvit-seg-b3-cityscapes": (
        partial(efficientvit_seg_b3, dataset="cityscapes"),
        1e-5,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_b3_cityscapes.pt",
    ),
    ############################################################################
    "efficientvit-seg-l1-cityscapes": (
        partial(efficientvit_seg_l1, dataset="cityscapes"),
        1e-7,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_l1_cityscapes.pt",
    ),
    "efficientvit-seg-l2-cityscapes": (
        partial(efficientvit_seg_l2, dataset="cityscapes"),
        1e-7,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_l2_cityscapes.pt",
    ),
    ############################################################################
    "efficientvit-seg-b1-ade20k": (
        partial(efficientvit_seg_b1, dataset="ade20k"),
        1e-5,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_b1_ade20k.pt",
    ),
    "efficientvit-seg-b2-ade20k": (
        partial(efficientvit_seg_b2, dataset="ade20k"),
        1e-5,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_b2_ade20k.pt",
    ),
    "efficientvit-seg-b3-ade20k": (
        partial(efficientvit_seg_b3, dataset="ade20k"),
        1e-5,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_b3_ade20k.pt",
    ),
    ############################################################################
    "efficientvit-seg-l1-ade20k": (
        partial(efficientvit_seg_l1, dataset="ade20k"),
        1e-7,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_l1_ade20k.pt",
    ),
    "efficientvit-seg-l2-ade20k": (
        partial(efficientvit_seg_l2, dataset="ade20k"),
        1e-7,
        "assets/checkpoints/efficientvit_seg/efficientvit_seg_l2_ade20k.pt",
    ),
}


def create_efficientvit_seg_model(
    name: str,
    pretrained=True,
    weight_url: Optional[str] = None,
    pretrained_strict: Optional[bool] = None,
    **kwargs,
) -> EfficientViTSeg:
    if name not in REGISTERED_EFFICIENTVIT_SEG_MODEL:
        raise ValueError(
            f"Cannot find {name} in the model zoo. List of models: {list(REGISTERED_EFFICIENTVIT_SEG_MODEL.keys())}"
        )
    else:
        model_cls, norm_eps, default_pt = REGISTERED_EFFICIENTVIT_SEG_MODEL[name]
        model = model_cls(**kwargs)
        set_norm_eps(model, norm_eps)
        weight_url = default_pt if weight_url is None else weight_url

    if pretrained:
        if weight_url is None:
            raise ValueError(f"Cannot find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            if pretrained_strict is None:
                pretrained_strict = (
                    kwargs.get("attention_type", "lite_mla") == "lite_mla"
                    and kwargs.get("use_multiscale_qkv", True)
                )
            if pretrained_strict:
                model.load_state_dict(weight)
                model.pretrained_load_report = {
                    "loaded": len(weight),
                    "missing": [],
                    "unexpected": [],
                    "shape_mismatch": [],
                }
            else:
                model_state = model.state_dict()
                compatible = {
                    key: value
                    for key, value in weight.items()
                    if key in model_state and model_state[key].shape == value.shape
                }
                shape_mismatch = sorted(
                    key for key, value in weight.items() if key in model_state and model_state[key].shape != value.shape
                )
                incompatible = model.load_state_dict(compatible, strict=False)
                model.pretrained_load_report = {
                    "loaded": len(compatible),
                    "missing": sorted(incompatible.missing_keys),
                    "unexpected": sorted(key for key in weight if key not in model_state),
                    "shape_mismatch": shape_mismatch,
                }
    return model


def create_efficientvit_seg_experiment_model(
    name: str,
    experiment: str,
    mhla_num_groups: int = 4,
    **kwargs,
) -> EfficientViTSeg:
    """Build one of the controlled E0--E3 attention variants."""
    experiment = experiment.lower()
    if experiment not in EFFICIENTVIT_SEG_EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; choose from {list(EFFICIENTVIT_SEG_EXPERIMENTS)}")
    variant_kwargs = dict(EFFICIENTVIT_SEG_EXPERIMENTS[experiment])
    variant_kwargs["mhla_num_groups"] = mhla_num_groups
    variant_kwargs.update(kwargs)
    return create_efficientvit_seg_model(name=name, **variant_kwargs)
