import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from calculate_coref_distance import calculate_coref_distances

folder = Path("../gpt2_coref_results")

gap_distances: list[int] = []
for file in folder.glob("*.conllu"):
    df = calculate_coref_distances(file)
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
plt.xticks(np.arange(min_val, max_val + 1, 100))

plt.xlabel("Coreference Distance")
plt.ylabel("Density")
plt.savefig("../result/fig/coref_dist_dist.png", dpi=600, bbox_inches='tight')
