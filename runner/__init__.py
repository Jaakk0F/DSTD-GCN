from .amass import AMASSRunner
from .cmu import CMURunner
from .expi import EXPIRunner
from .h36m import H36MRunner
from .pw3d import PW3DRunner


def get_runner(runner_type, runner_config):
    return {
        "h36m": H36MRunner,
        "cmu": CMURunner,
        "amass": AMASSRunner,
        "expi": EXPIRunner,
        "3dpw": PW3DRunner,
    }[runner_type](runner_config)
