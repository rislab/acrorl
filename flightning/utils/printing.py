from jax import numpy as jnp


def print_metrics(metrics, prefix=""):
    def fmt(x, scale=1.0):
        return f"{float(x) * scale:.4f}"

    labels = {
        "pos_rmse": "Position RMSE",
        "settle_time": "Settling Time",
        "max_x_dev": "Max X Position Deviation",
        "max_y_dev": "Max Y Position Deviation",
        "max_z_dev": "Max Z Position Deviation",
    }
    units = {
        "pos_rmse": "m",
        "settle_time": "s",
        "max_x_dev": "m",
        "max_y_dev": "m",
        "max_z_dev": "m",
    }

    print(f"{prefix}Performance Metrics  (n={metrics['num_trajs']})")
    print("-" * 78)
    header = f"{'Metric':<26}{'mean':>9}{'std':>9}{'median':>9}{'min':>9}{'max':>9}{'95% CI':>17}"
    print(header)
    print("-" * 78)

    for key, label in labels.items():
        s = metrics["summary"][key]
        ci = f"[{fmt(s['ci95_lower'])}, {fmt(s['ci95_upper'])}]"
        print(
            f"{label:<26}"
            f"{fmt(s['mean']):>9}"
            f"{fmt(s['std']):>9}"
            f"{fmt(s['median']):>9}"
            f"{fmt(s['min']):>9}"
            f"{fmt(s['max']):>9}"
            f"{ci:>17}"
            f"  {units[key]}"
        )

    print("-" * 78)
    print(f"Success Rate              : {fmt(metrics['success_rate'], 100)} %")
    print(f"Failure Count             : {metrics['failure_count']} / {metrics['num_trajs']}")
    print(f"Crash Rate                : {fmt(metrics['crash_rate'], 100)} %")
    print("-" * 78)