import os
import json
from pykeen.pipeline import pipeline
from util_eval_pykeen import plot_loss, get_metrics
import pandas as pd
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

save_dir = 'pykeen_eval/test_pre_stratified/'
model_list = ["TransE", "ComplEx", "TransH", "DistMult", "ProjE"] # "ConvE"
n_epochs = [200, 500, 1000, 2000, 5000]

df = []

for model_name in model_list:
    for epochs in n_epochs:
        print(f"train and eval {model_name} with {epochs} epochs")
        save_path = os.path.join(save_dir, f"{model_name}{epochs}")
        result = pipeline(
            training="data/train.tsv",
            validation="data/validate.tsv",
            testing="data/test.tsv",
            model=model_name,
            epochs=epochs,
            device=device
        )
        result.save_to_directory(save_path)

        with open(os.path.join(save_path, "results.json"), "rb") as f:
            results = json.load(f)
        plot_loss(results["losses"], save_path)
        metrics = get_metrics(results)
        for k, v in metrics.items():
            if k.startswith("hits_at_"):
                metrics[k] = v * 100
        metrics["model"] = model_name
        metrics["epochs"] = epochs

        df.append(metrics)

df = pd.DataFrame(df)
df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)