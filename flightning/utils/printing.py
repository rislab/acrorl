from jax import numpy as jnp


def print_metrics(metrics, prefix=""):
    def fmt(x, scale=1.0):
        return f"{float(x) * scale:.4f}"

    print(f"{prefix}Performance Metrics")
    print("-" * 50)
    print(f"Position RMSE            : {fmt(metrics['pos_rmse_mean'])} m")
    print(f"Settling Time (Inversion): {fmt(metrics['settle_time_mean'])} s")
    print(f"Max X Position Deviation : {fmt(metrics['max_x_dev'])} m")
    print(f"Max Y Position Deviation : {fmt(metrics['max_y_dev'])} m")
    print(f"Max Z Position Deviation : {fmt(metrics['max_z_dev'])} m")
    print(f"Success Rate             : {fmt(metrics['success_rate'], 100)} %")
    print("-" * 50)
