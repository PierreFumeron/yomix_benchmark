import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
# Code for supplementary figure comparing performances of KNN SVM RF

# datasets = ["citeseq", "meth", "lawlor", "pbmc", "tcga", "proteomics_nonan"]
datasets = ["pbmc", "proteomics_nonan", "sarc_ba", "recount_log_normalized_hvg", "citeseq"]
# Directory where the dataset CSV files are located
result_dir = "result"

features_to_include = [1, 3, 5, 10, 15, 20]

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
method_names = {
    'yomix': "Yomix",
    'cosg': "Cosg",
    'scanpy_wilcoxon': "Scanpy wilcoxon",
    'scanpy_t-test': "Scanpy t-test"
}

model_names = {
    "knn": "KNN",
    "rf": "Random Forest",
    "svm": "SVM"
}

all_data = []
for dataset in datasets:
    file_path = os.path.join(result_dir, f"{dataset}.csv")
    if not os.path.exists(file_path):
        print(f"Warning: file {file_path} not found, skipping.")
        continue
    
    df = pd.read_csv(file_path, index_col=0)
    df['dataset'] = dataset 
    all_data.append(df)


if not all_data:
    print("No data was loaded. Please check the 'result_dir' path and file names.")
else:
    combined_df = pd.concat(all_data)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    for ax, clf in zip(axes, model_names.keys()):

        filtered_df = combined_df[combined_df['nb_genes'].isin(features_to_include)]
        filtered_df = filtered_df.loc[filtered_df['model']==clf]
        filtered_df['method'] = filtered_df['method'].map(method_names)
        
        sns.pointplot(
            data=filtered_df,
            x="nb_genes",
            y="mcc",
            hue="method",
            palette="colorblind",
            markers=["o", "s", "D", "v", "^", "<", ">"],
            linestyles=["-", "--", "-.", ":", "-", "--", "-."],
            errorbar="sd", 
            capsize=.1,
            ax=ax
        )

        ax.set_title(f"{model_names[clf]}", weight="bold")
        ax.set_xlabel("Number of Top Features")
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
        ax.legend_.remove()

        if ax != axes[0]:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Matthews Correlation Coefficient (MCC)")


    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="best",
        ncol=2,
        title="Method",
        frameon=True,
        facecolor='white',
        edgecolor='black',
        framealpha=1
    )

        # --- Save the Plot ---
    fig.suptitle("Average Performance Across All Datasets", weight="bold")

    plt.tight_layout()
    plt.savefig("plots/classifier_comparison_all_datasets.png")
    plt.show()
        # print("Plot saved as 'average_comparison_all_datasets.svg'")