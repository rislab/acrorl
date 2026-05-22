import jax
from jax import numpy as jnp
from typing import List
from .base_trajectory import BaseTrajectory


@jax.tree_util.register_pytree_node_class
class CompositeTrajectory:
    def __init__(
        self,
        trajectories: List[BaseTrajectory],
    ):
        """Function must take f(time, duration, *args)"""
        time = 0
        self.T = tuple([(time := time + traj.length()) for traj in trajectories])
        self.trajs = tuple(trajectories)

    def get_reference(self, time: float):
        return self.__call__(time)

    def __call__(self, time: float):
        T_arr = jnp.asarray(self.T)

        idx = jnp.argmax(T_arr >= time)
        idx = jnp.clip(idx, 0, len(self.trajs) - 1)

        prev_time = jax.lax.cond(
            idx > 0,
            lambda _: T_arr[idx - 1],
            lambda _: jnp.array(0.0),
            operand=None,
        )

        local_time = time - prev_time

        def make_branch(i):
            def branch(t):
                return self.trajs[i].get_reference(t)

            return branch

        branches = tuple(make_branch(i) for i in range(len(self.trajs)))

        return jax.lax.switch(idx, branches, local_time)

    def batched(self, times):
        f = lambda ti: self.__call__(ti)
        return jax.vmap(f)(times)

    def tree_flatten(self):
        children, treedef = jax.tree_util.tree_flatten(self.trajs)
        aux = (treedef, self.T)
        return tuple(children), aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        treedef, T = aux
        trajs = jax.tree_util.tree_unflatten(treedef, list(children))
        obj = cls.__new__(cls)
        obj.trajs = tuple(trajs)
        obj.T = tuple(T)
        return obj
