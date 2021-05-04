from torch.nn.modules.loss import *   # noqa: F401,F403
from .cross_entropy_smooth_loss import CrossEntropySmoothLoss
from .cross_entropy_semantic_loss import CrossEntropySemanticLoss
from .coding_loss import CodingLoss
from .kl_loss import KLLoss
from .wsl_loss import WSLLoss
from .wslv2_loss import WSLv2Loss