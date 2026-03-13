import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path


dataset_info = {
    'citeseq': {'samples': 8617, 'features': 2000},
    'pbmc':    {'samples': 2638, 'features': 13714},
    'sarc_ba': {'samples': 1077,  'features': 428230},
    'proteomics_nonan':    {'samples': 1549,'features': 3786},
    'recount_log_normalized_hvg': {'samples': 30000, 'features':8000}
}

mapping_names = {
    'citeseq':'CITE-Seq',
    'pbmc': "PBMC",
    'sarc_ba': "DNA Methylation",
    'proteomics_nonan': "Proteomics",
    'recount_log_normalized_hvg': 'TCGA + GTEX'
}

method_names = {
    'yomix': "Yomix",
    'cosg': "Cosg",
    'scanpy_wilcoxon': "wilcoxon",
    'scanpy_t-test': "t-test"
}

method_colors = {
    'Yomix': '#0173B2',
    'Cosg': '#DE8F05',
    'wilcoxon': '#029E73',
    't-test': '#D55E00'
}

result_dir = "result" # Update this path as needed

if not os.path.isdir(result_dir):
    print(f"Error: The specified directory does not exist: {result_dir}")
    
file_names = [fn for fn in os.listdir(result_dir) if fn.endswith("_runtime.csv") ]
if not file_names:
    print(f"No '_runtime.csv' files found in '{result_dir}'.")
    # return

all_dfs = []
for file_name in file_names:
    df_tmp = pd.read_csv(os.path.join(result_dir, file_name), index_col=0)
    df_long = df_tmp.reset_index().melt(id_vars='index', var_name='run', value_name='time')
    df_long = df_long.rename(columns={'index': 'method'})
    df_long['method'] = df_long['method'].map(method_names)
    df_long["dataset"] = file_name.replace("_runtime.csv", "")
    all_dfs.append(df_long)
    
df_all_runtimes = pd.concat(all_dfs, ignore_index=True)



datasets = ['pbmc', 'sarc_ba', 'proteomics_nonan', 'citeseq','recount_log_normalized_hvg']
df_all_runtimes = df_all_runtimes[df_all_runtimes['dataset'].isin(datasets)]
def remove_scanpy(x):
    if "scanpy" in x.lower():
        return x.split('Scanpy ')[1]
    return x
df_all_runtimes['method'] = df_all_runtimes['method'].apply(remove_scanpy)
tmp = df_all_runtimes.copy(deep=True)
wilcoxon_values_per_dataset = df_all_runtimes[df_all_runtimes['method']=="wilcoxon"].groupby(["dataset"])['time'].mean().to_dict()
for dataset in datasets:
    tmp.loc[tmp['dataset']==dataset, 'time'] = df_all_runtimes.loc[df_all_runtimes['dataset']==dataset, 'time']/wilcoxon_values_per_dataset[dataset]


sns.set_theme(style="whitegrid", context="paper", font_scale=1.7)
plt.figure(figsize=(14, 8)) 

# REORDER THE LIST 
dataset_order = ['sarc_ba', 'recount_log_normalized_hvg', 'citeseq', 'pbmc',  'proteomics_nonan']

ax = sns.barplot(
    data=tmp,
    x="dataset",
    y="time",
    hue="method",
    palette=method_colors,
    errorbar="sd",
    order=dataset_order,  # This line uses the new order,
    err_kws={'linewidth':3}
)

# ax.set_xlabel("Dataset", fontsize=20)# , weight='bold')
ax.set_ylabel("Average Normalized Runtime", fontsize=20)# , weight='bold')
ax.set_title("Average Runtime Across Datasets", fontsize=24, weight='bold')

positions = ax.get_xticks()
new_labels = []
current_labels = [label.get_text() for label in ax.get_xticklabels()]
for dataset_name in current_labels:
    info = dataset_info.get(dataset_name, {'samples': '?', 'features': '?'})
    display_name = mapping_names[dataset_name]
    new_labels.append(
        f"{display_name}\n{info['samples']}\n{info['features']}"
    )

ax.set_xticks(positions)
ax.set_xticklabels(new_labels)
plt.ylim((0.0,1.4))
plt.xticks(rotation=0, ha='center', fontsize=18)
plt.yticks(fontsize=18)

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles, labels,
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(.5, 0.89),
    # # title='Method',
    # loc='upper left',
    # title_fontsize=22,
    fontsize=19,
    frameon=True,
    facecolor='white',
    edgecolor='black',
    framealpha=1
)

plt.tight_layout()
# I've changed the output filename to reflect the new order
# plt.savefig('runtime_comparison_tcga_last.svg', dpi=1200)
plt.savefig('plots/runtime_wilcoxon.svg')
plt.show()