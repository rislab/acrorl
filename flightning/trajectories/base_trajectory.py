import jax
from flightning.objects import QuadrotorState


@jax.tree_util.register_pytree_node_class
class BaseTrajectory:
    def __init__(
        self,
        init_quadrotor_state: QuadrotorState,
        init_yaw: jax.Array,
        eta: jax.Array,
        duration: float,
    ):
        self.duration = float(duration)
        self.init_quadrotor_state = init_quadrotor_state
        self.init_yaw = init_yaw
        self.eta = eta

    def get_reference(self, time):
        return self.__call__(time)

    def __call__(self, time):
        raise NotImplementedError

    def length(self):
        return self.duration

    def tree_flatten(self):
        children = (self.init_quadrotor_state, self.init_yaw, self.eta)
        aux = self.duration
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        init_quadrotor_state, init_yaw, eta = children
        obj = cls.__new__(cls)
        obj.duration = aux
        obj.init_quadrotor_state = init_quadrotor_state
        obj.init_yaw = init_yaw
        obj.eta = eta
        return obj
