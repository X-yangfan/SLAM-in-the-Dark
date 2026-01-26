__all__ = ["DispResNet", "PoseResNet", "ResnetEncoder"]


def __getattr__(name: str):
    if name == "DispResNet":
        from .disp_resnet import DispResNet

        return DispResNet
    if name == "PoseResNet":
        from .pose_resnet import PoseResNet

        return PoseResNet
    if name == "ResnetEncoder":
        from .resnet_encoder import ResnetEncoder

        return ResnetEncoder
    raise AttributeError(name)
