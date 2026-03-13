from tqdm import tqdm
from yomix_signature import compute_signature
from scipy.sparse import issparse
import numpy as np
import scanpy as sc
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


file = "tcga_half_raw_deg_hvg"
xd = sc.read_h5ad(f"pydeseq2_analysis/{file}.h5ad")

def _to_dense(x):
    if issparse(x):
        return x.todense()
    else:
        return x
xd.X = np.asarray(_to_dense(xd.X))
min_norm = np.min(xd.X, axis=0)
max_norm = np.max(xd.X, axis=0)
xd.X = np.divide(xd.X - min_norm, max_norm - min_norm + 1e-8)

def var_mean_values(adata) -> np.ndarray:
    return np.squeeze(np.asarray(np.mean(adata.X, axis=0)))

def var_standard_deviations(adata) -> np.ndarray:
    return np.squeeze(np.asarray(np.std(adata.X, axis=0)))

xd.var["mean_values"] = var_mean_values(xd)
xd.var["standard_deviations"] = var_standard_deviations(xd)


labels = xd.obs["labels"].unique()
benchmarks = [(label, "rest") for label in labels]

ranked_genes = {}

xd.obs["id"] = [i for i in range(xd.obs.shape[0])]
for label_a, label_b in tqdm(benchmarks):

    signature_key = str(label_a) + "_vs_" + str(label_b)
    ranked_genes[signature_key] = {}
    xd.obs["binary_labels"] = np.where(
                xd.obs["labels"] == label_a, label_a, "rest"
    )
    xd.obs["binary_labels"] = pd.Categorical(
        xd.obs["binary_labels"], categories=[str(label_a), "rest"]
    )
    xd = xd[~xd.obs['binary_labels'].isna()]
    indices_label = xd[xd.obs["labels"] == label_a, :].obs["id"].to_list()
    genes, _, _ = compute_signature(
        adata=xd,
        means=xd.var["mean_values"],
        stds=xd.var["standard_deviations"],
        obs_indices_A=indices_label,
    )
    ranked_genes[signature_key]["yomix"] = xd.var.iloc[genes].index.tolist()


comparisons = list(ranked_genes.keys())
n_row=7
n_col=5
fig, axes = plt.subplots(n_row, n_col, figsize=(13, 15), )
axes = axes.flatten()

for i, col in enumerate(comparisons):
    ax = axes[i]

    tmp_df = xd.var.copy()
    tmp_df["-logpvalue"] = -np.log10(xd.var[col.split("_vs")[0] + "_pvalue"])
    tmp_df['yomix'] = [gene in ranked_genes[col]["yomix"] for gene in xd.var_names]

    scatter = sns.scatterplot(
        data=tmp_df,
        x=col.split("_vs")[0] + "_log2FoldChange",
        y="-logpvalue",
        hue="yomix",
        ax=ax
    )
    ax.get_legend().remove()

    # store legend info from the first plot
    if i==1:
        legend_handles, legend_labels = scatter.get_legend_handles_labels()

    ax.set_title(col.split('_')[1])
    
    
    # keep y_label for first column only
    if i%n_col==0:
        ax.set_ylabel("-log(pvalue)")
    else:
        ax.set_ylabel("")
    # keep x_label for last row
    if (i+2)//n_col == n_row-1:
        ax.set_xlabel("Log2FoldChange")
    else:
        ax.set_xlabel("")

axes[33].axis("off")
axes[34].axis("off")
axes[33].legend(legend_handles, legend_labels, title="yomix", loc="center")

# plt.title("Volcano plots using pydeseq2 on TCGA")
plt.tight_layout()
# plt.show()
plt.savefig(f'plots/volcano_plots_{file}.png')