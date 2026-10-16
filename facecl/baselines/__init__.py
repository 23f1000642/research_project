from .icarl import ICaRL
from .regularization import MAS, SI, EWCOnline, LwF
from .replay import ER, MIR, SGD, DERpp

BASELINES = {
    "sgd": SGD,
    "joint": SGD,  # the trainer feeds all tasks at once
    "er": ER,
    "derpp": DERpp,
    "mir": MIR,
    "ewc_on": EWCOnline,
    "si": SI,
    "mas": MAS,
    "lwf": LwF,
    "icarl": ICaRL,
}


def build_method(name, model, bench, cfg, device):
    from ..methods import FACE

    registry = dict(BASELINES, face=FACE, face_pr=FACE)
    if name not in registry:
        raise ValueError(f"unknown method {name}; options: {sorted(registry)}")
    return registry[name](model, bench, cfg, device)
