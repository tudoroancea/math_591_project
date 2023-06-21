from .system_models import *
from .control_models import *
from .mlp import *

# ode_dims = {
#     "kin4": (Kin4, KIN4_NXTILDE, KIN4_NUTILDE),
#     "dyn6": (Dyn6, DYN6_NXTILDE, DYN6_NUTILDE),
#     "neural_dyn6": (
#         NeuralDyn6,
#         DYN6_NXTILDE,
#         DYN6_NUTILDE,
#         DYN6_NXTILDE + DYN6_NUTILDE - 3,
#         DYN6_NXTILDE - 3,
#     ),
# }

dt = 1 / 20
