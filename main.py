import argparse
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from yomix_signature import compute_signature
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.svm import SVC
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import cosg
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def main(
    xd,
    output_filename,
    label_column="labels",
    nb_clf_runs=5,  # Number of runs for the classifier
    signatures_size=[1, 3, 5, 10, 15, 20],
):
    """
    Runs benchmarking of feature selection methods.
    This function compares different marker selection methods
    (yomix, cosg, scanpy_wilcoxon, scanpy_t-test)
    and evaluates their performance using a classifier
    (default: SVM) across various gene signature sizes.
    Parameters
    ----------
    xd : AnnData
        Annotated data matrix (e.g., from Scanpy) containing gene
        expression data and cell metadata.
    output_filename : str
        Name of the output CSV file (without extension) to save
        benchmarking results.
    label_column : str
        Column in `xd.obs` containing cell type or class labels.
    nb_clf_runs : int, optional
        Number of classifier runs with different random seeds (default is 10).
    Returns
    -------
    ranked_genes : dict
        Dictionary containing ranked gene lists for each method and comparison."""
    
    xd.obs["id"] = [i for i in range(xd.obs.shape[0])]
    all_methods = ["yomix", "cosg", "wilcoxon", "t-test"]
    runtime = {}
    results = []
    ranked_genes = {}
    top_genes_table=[]
    xd.obs[label_column] = xd.obs[label_column].astype(str)

    signatures_size = sorted(signatures_size)
    labels = xd.obs[label_column].unique()
    # define rest here what is insdie rest
    labels = [label for label in labels if label != "rest"]
    benchmarks = [(label, "rest") for label in labels]

    for label_a, label_b in tqdm(benchmarks):
        signature_key = str(label_a) + "_vs_" + str(label_b)
        runtime[signature_key] = {}
        ranked_genes[signature_key] = {}
        # print(f"\n Comparing: {label_a} vs {label_b}")
        xd.obs["binary_labels"] = np.where(
            xd.obs[label_column] == label_a, label_a, "rest"
        )
        xd.obs["binary_labels"] = pd.Categorical(
            xd.obs["binary_labels"], categories=[str(label_a), "rest"]
        )
        start_time = time.time()

        cosg.cosg(
            xd,
            key_added="cosg",
            use_raw=False,
            mu=100,
            expressed_pct=0.05,
            remove_lowly_expressed=True,
            n_genes_user=100,
            groupby="binary_labels",
        )
        runtime[signature_key]["cosg"] = time.time() - start_time
        ranked_genes[signature_key]["cosg"] = pd.DataFrame(
            xd.uns["cosg"]["names"], columns=xd.uns["cosg"]["names"].dtype.names
        )[str(label_a)].values
        # print(ranked_genes[signature_key]["cosg"])
        methods_scanpy = ["wilcoxon", "t-test"]

        for method_sc in methods_scanpy:
            start_time = time.time()
            sc.tl.rank_genes_groups(
                xd,
                groupby="binary_labels",
                groups=[label_a],
                reference=label_b,
                method=method_sc,
            )
            runtime[signature_key][method_sc] = time.time() - start_time
            ranked_genes[signature_key][method_sc] = xd.uns[
                "rank_genes_groups"
            ]["names"][label_a]

        indices_label = xd[xd.obs[label_column] == label_a, :].obs["id"].to_list()
        start_time = time.time()
        genes, _, _ = compute_signature(
            adata=xd,
            means=xd.var["mean_values"],
            stds=xd.var["standard_deviations"],
            obs_indices_A=indices_label,
        )
        runtime[signature_key]["yomix"] = time.time() - start_time
        ranked_genes[signature_key]["yomix"] = xd.var.iloc[genes].index.tolist()

        for method in all_methods:
            for size in signatures_size:
                selected_genes = ranked_genes[signature_key][method][:size]
                # save signature 
                if size == signatures_size[-1]:
                    top_genes_table.extend({
                        "method": method,
                        "comparison": signature_key,
                        "rank": nb,
                        "gene": gene
                    }
                    for nb, gene in enumerate(selected_genes, start=1)
                    )

                X_subset = xd[:, selected_genes].X
                X_subset = (
                    X_subset.toarray() if hasattr(X_subset, "toarray") else X_subset
                )
                y_binary = np.where(xd.obs.binary_labels == label_a, 1, 0)

                for run in range(nb_clf_runs):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_subset,
                        y_binary,
                        test_size=0.3,
                        stratify=y_binary,
                        random_state=run,
                    )


                    classifiers = {
                        "svm": SVC(
                            kernel="linear", class_weight="balanced", random_state=run
                        ),
                        "knn":KNeighborsClassifier(),
                        "rf": RandomForestClassifier(random_state=run)
                    }
                    for clf_name in classifiers:
                        classifiers[clf_name].fit(X_train, y_train)

                        y_pred = classifiers[clf_name].predict(X_test)
                        results.append(
                            {
                                "method": method,
                                "mcc": matthews_corrcoef(y_test, y_pred),
                                "precision": precision_score(y_test, y_pred),
                                "f1_score": f1_score(y_test, y_pred),
                                "recall": recall_score(y_test, y_pred),
                                "label_vs_rest": signature_key,
                                "nb_genes": size,
                                "model": clf_name,
                            }
                        )

    runtime_df = pd.DataFrame(runtime)
    # ensure result directory exists
    result_dir = Path("result")
    result_dir.mkdir(parents=True, exist_ok=True)
    runtime_df.to_csv(f"result/{output_filename}_runtime.csv")
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"result/{output_filename}.csv")
    genes_df=pd.DataFrame(top_genes_table)
    genes_df.to_csv(f"result/{output_filename}_top_genes.csv")

    return res_df, runtime, ranked_genes, genes_df


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Perform KNN, SVM and RF on yomix, cosg, t-test and wilcoxon signatures for ALL labels vs rest."
    )
    parser.add_argument(
        "file", type=str, nargs="?", default=None, help="the .ha5d file to open"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output h5ad file.",
    )
    return parser.parse_args()



if __name__ == "__main__":

    args = parse_arguments()
    filearg = Path(args.file)
    xd = sc.read_h5ad(filearg.absolute())
    output_filename = args.output
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

    results = main(
        xd,
        output_filename=output_filename,
    )

