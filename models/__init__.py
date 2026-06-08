# models/__init__.py
from .arl_unet import ARLUNet
from .emaunet import EMAUNet
from .smp_unet import SMPUNet
from .ucm_net import UCMNet


def build_model(config, logger=None):
    """
    Instantiate the segmentation model from `config.network` and `config.model_config`.
    """
    cfg = config.model_config
    network = getattr(config, "network", "emaunet")

    if network == "emaunet":
        return EMAUNet(
            num_classes=cfg["num_classes"],
            input_channels=cfg["input_channels"],
            c_list=cfg["c_list"],
            bridge=cfg["bridge"],
            deep_supervision=cfg["deep_supervision"],
            ff_parser_cfg=cfg.get("ff_parser_cfg"),
            logger=logger,
        )

    if network == "arl_unet":
        c_list = cfg.get("arl_c_list", cfg.get("c_list"))
        arl_defaults = {
            "use_artifact_stem": True,
            "use_robust_encoder": True,
            "use_skip_fusion": True,
            "skip_fusion_mode": "soft",
        }
        arl_cfg = {**arl_defaults, **(cfg.get("arl_cfg") or {})}
        return ARLUNet(
            num_classes=cfg["num_classes"],
            input_channels=cfg["input_channels"],
            c_list=c_list,
            deep_supervision=cfg["deep_supervision"],
            ff_parser_cfg=cfg.get("ff_parser_cfg"),
            arl_cfg=arl_cfg,
            logger=logger,
        )

    if network == "unet":
        return SMPUNet(
            encoder_name=cfg.get("smp_encoder", "resnet34"),
            in_channels=cfg["input_channels"],
            classes=cfg["num_classes"],
            logger=logger,
        )

    if network == "ucm_net":
        embed_dims = cfg.get("c_list", [8, 16, 24, 32, 48, 64]) + [cfg["input_channels"]]
        return UCMNet(
            num_classes=cfg["num_classes"],
            input_channels=cfg["input_channels"],
            deep_supervision=cfg["deep_supervision"],
            img_size=cfg.get("img_size", 256),
            embed_dims=embed_dims,
            logger=logger,
        )

    raise ValueError(f"Unsupported network: {network}")
