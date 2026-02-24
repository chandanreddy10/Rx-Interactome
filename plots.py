import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

# medgemma_fine_tuned = "medgemma-4b-it-sft-lora-interactome/checkpoint-678/trainer_state.json"
tx_Gemma_fine_tuned="txgemma_finetuned/checkpoint-444/trainer_state.json"
with open(tx_Gemma_fine_tuned, "r") as file:
    log_history = json.load(file)["log_history"]

os.makedirs("plots", exist_ok=True)
data = []

for entry in log_history:
    step = entry.get("step")
    if step is None:
        continue

    if "loss" in entry:
        data.append({
            "step": step,
            "value": entry["loss"],
            "metric": "Training Loss"
        })

    if "eval_loss" in entry:
        data.append({
            "step": step,
            "value": entry["eval_loss"],
            "metric": "Evaluation Loss"
        })

    if "mean_token_accuracy" in entry:
        data.append({
            "step": step,
            "value": entry["mean_token_accuracy"],
            "metric": "Mean Token Accuracy"
        })

df = pd.DataFrame(data)

sns.set_theme(style="whitegrid")

loss_metrics = ["Training Loss", "Evaluation Loss", "Mean Token Accuracy"]

for metric in df["metric"].unique():
    subset = df[df["metric"] == metric].sort_values("step")

    if metric in loss_metrics:
        subset["smoothed_value"] = subset["value"].rolling(window=50, min_periods=1).mean()
        y_column = "smoothed_value"
        title_suffix = ""
    else:
        y_column = "value"
        title_suffix = ""

    plt.figure()
    sns.lineplot(data=subset, x="step", y=y_column, marker="o")
    plt.title(f"{metric}{title_suffix}")
    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.tight_layout()

    # filename = metric.lower().replace(" ", "_") + ".png"
    filename = metric.lower().replace(" ", "_") + "tx_gemma"+".png"
    plt.savefig(f"plots/{filename}", dpi=300)

    plt.close()