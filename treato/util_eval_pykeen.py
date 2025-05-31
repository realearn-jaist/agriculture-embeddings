import os
import matplotlib.pyplot as plt

def plot_loss(loss, metrics_path):
    plt.figure()
    plt.plot(loss, label="training loss")
    plt.legend()
    plt.savefig(os.path.join(metrics_path, "loss.png"))
    plt.clf()

# metrics_names must be in results.json file from pykeen
metrics_names = [
    "arithmetic_mean_rank",
    "inverse_arithmetic_mean_rank",
    "hits_at_1",
    "hits_at_3",
    "hits_at_5",
    "hits_at_10"
]
def print_metrics(results, metrics_names=metrics_names):
    print(*metrics_names, sep="\t")
    print(*[round(float(results["metrics"]["both"]["realistic"][m]), 4) for m in metrics_names], sep="\t")

def get_metrics(results, metrics_names=metrics_names):
    return {m:results["metrics"]["both"]["realistic"][m] for m in metrics_names}
    