from .cmu import CMUMocap

from .h36m import Human36M
from .pw3d import PW3D
from .utils import define_actions, define_actions_cmu


def get_dataset(dataset_type, **dataset_opts):
    return {
        "h36m": Human36M,
        "cmu": CMUMocap,
        "3dpw": PW3D,
    }[dataset_type](**dataset_opts[dataset_type])
