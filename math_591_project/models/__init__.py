from .control_models import *
from .mlp import *

ode_dims = {
    "kin4": (Kin4ODE, KIN4_NXTILDE, KIN4_NUTILDE),
    "dyn6": (Dyn6ODE, DYN6_NXTILDE, DYN6_NUTILDE),
    "blackbox_kin4": (
        BlackboxKin4ODE,
        KIN4_NXTILDE,
        KIN4_NUTILDE,
        KIN4_NXTILDE + KIN4_NUTILDE,
        KIN4_NXTILDE,
    ),
    "blackbox_dyn6": (
        BlackboxDyn6ODE,
        DYN6_NXTILDE,
        DYN6_NUTILDE,
        DYN6_NXTILDE + DYN6_NUTILDE - 3,
        DYN6_NXTILDE - 3,
    ),
}


