
def get_artifact_curriculum_policy(epoch: int):
    """
    Three-stage artifact curriculum policy (research-report2.md).

    epoch: 1-based epoch index.
    """
    if epoch <= 160:
        return {
            "stage": "Stage 1",
            "enabled": False,
            "p": 0.0,
            "artifacts": (),
            "severities": (),
            "lr_scale": 1.0,
        }
    if epoch <= 240:
        return {
            "stage": "Stage 2",
            "enabled": True,
            "p": 0.08,
            "artifacts": ("hair", "skin_line", "air_bubble", "highlight"),
            "severities": (1,),
            "lr_scale": 1.0,
        }
    return {
        "stage": "Stage 3",
        "enabled": True,
        "p": 0.12,
        "artifacts": ("hair", "skin_line", "air_bubble", "highlight"),
        "severities": (1, 2),
        "lr_scale": 1.0,
    }


def get_artifact_curriculum_lr_scale(epoch: int) -> float:
    return get_artifact_curriculum_policy(epoch)["lr_scale"]
