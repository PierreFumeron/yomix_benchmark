import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# datasets = ["citeseq", "meth", "lawlor", "pbmc", "tcga", "proteomics_nonan"]
datasets = ["pbmc", "proteomics_nonan", "sarc_ba", "tcga", "recount_log_normalized_hvg"]
# Directory where the dataset CSV files are located
# IMPORTANT: Please update this path to the correct location of your files.
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

all_data = []
for dataset in datasets:
    file_path = os.path.join(result_dir, f"{dataset}.csv")
    if not os.path.exists(file_path):
        print(f"Warning: file {file_path} not found, skipping.")
        continue
    
    df = pd.read_csv(file_path, index_col=0)
    df['dataset'] = dataset 
    all_data.append(df)

model = "svm"
if not all_data:
    print("No data was loaded. Please check the 'result_dir' path and file names.")
else:
    combined_df = pd.concat(all_data)
    
    
    filtered_df = combined_df[combined_df['nb_genes'].isin(features_to_include)]
    filtered_df['method'] = filtered_df['method'].map(method_names)
    filtered_df = filtered_df[filtered_df["model"]==model]
    print(filtered_df.head())
    plt.figure(figsize=(14, 8))
    ax = sns.pointplot(
        data=filtered_df,
        x="nb_genes",
        y="mcc",
        hue="method",
        palette="colorblind",
        markers=["o", "s", "D", "v", "^", "<", ">"],
        linestyles=["-", "--", "-.", ":", "-", "--", "-."],
        errorbar="sd",  # Display standard deviation as error bars
        capsize=.1
    )

    plt.title("SVM Average Performance Across All Datasets", fontsize=24, weight="bold")
    plt.xlabel("Number of Top Features", fontsize=22)# , weight="bold")
    plt.ylabel("Matthews Correlation Coefficient (MCC)", fontsize=22)# , weight="bold")
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="best",
        ncol=2,
        fontsize=19,
        frameon=True,
        facecolor='white',
        edgecolor='black',
        framealpha=1
    )

    # --- Save the Plot ---
    plt.tight_layout()

    plt.savefig("plots/model_all_datasets.svg", dpi=1200)
    plt.show()
    # print("Plot saved as 'average_comparison_all_datasets.svg'")