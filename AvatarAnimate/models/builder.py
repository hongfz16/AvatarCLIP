from .pose_generation import (
    PoseOptimizer,
    VPoserOptimizer,
    VPoserRealNVP,
    VPoserCodebook
)
from .motion_generation import (
    MotionInterpolation,
    MotionOptimizer
)


def build_pose_generator(conf: dict):
    name = conf.pop('type')
    model_cls = {
        'PoseOptimizer': PoseOptimizer,
        'VPoserOptimizer': VPoserOptimizer,
        'VPoserRealNVP': VPoserRealNVP,
        'VPoserCodebook': VPoserCodebook
    }[name]
    model = model_cls(name=name, **conf)
    return model


def build_motion_generator(conf: dict):
    name = conf.pop('type')
    model_cls = {
        'MotionInterpolation': MotionInterpolation,
        'MotionOptimizer': MotionOptimizer
    }[name]
    model = model_cls(name=name, **conf)
    return model
