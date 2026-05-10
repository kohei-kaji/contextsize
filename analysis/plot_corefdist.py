import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from calculate_coref_distance import calculate_coref_distances

folder = Path("../gpt2_coref_results")

for dataset in ["ns", "brown", "provo", "os"]:
    folder_ = folder / dataset

    gap_distances: list[int] = []
    for file in folder_.glob("*.conllu"):
        df = calculate_coref_distances(file)
        if df.empty:
            continue
        gap_distances.extend(df["gap_distance"].tolist())


    plt.figure(figsize=(8,6))

    sns.histplot(
        gap_distances, 
        stat="density", 
        bins="auto",
        color="#63666A",
        edgecolor="white"
    )

    sns.kdeplot(
        gap_distances, 
        color="#041E42",
        linewidth=3,
        clip=(0, None)
    )

    min_val = 0
    max_val = max(gap_distances)
    if max_val > 800:
        plt.xticks(np.arange(min_val, max_val + 1, 100))
    else:
        plt.xticks(np.arange(min_val, max_val + 1, 10))

    match dataset:
        case "ns":
            plt.title("Natural Stories")
        case "brown":
            plt.title("Brown")
        case "provo":
            plt.title("Provo")
        case "os":
            plt.title("OneStop")
    plt.xlabel("Coreference Distance")
    plt.ylabel("Density")
    plt.savefig(f"../result/fig/coref_dist_{dataset}_dist.png", dpi=600, bbox_inches='tight')
