from .cmu import CMURunner
from .h36m import H36MRunner
from .pw3d import PW3DRunner


def get_runner(runner_type, runner_config):
    return {
        "h36m": H36MRunner,
        "cmu": CMURunner,
        "3dpw": PW3DRunner,
    }[runner_type](runner_config)
