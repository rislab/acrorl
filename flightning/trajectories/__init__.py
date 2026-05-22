from .base_trajectory import BaseTrajectory
from .composite_trajectory import CompositeTrajectory
from .constant_trajectory import ConstantTrajectory
from .loop_trajectory import LoopTrajectory
from .polynomial_trajectory import PolynomialTrajectory

TRAJECTORY_REGISTRY = {
    "constant": ConstantTrajectory,
    "loop": LoopTrajectory,
    "polynomial": PolynomialTrajectory,
}
