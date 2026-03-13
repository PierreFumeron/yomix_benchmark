#!/usr/bin/env python3
"""
Differential expression analysis using PyDESeq2 for ALL conditions vs rest.
 
This script:
1. Detects all unique conditions from condition_column
2. Runs PyDESeq2 for each condition vs rest
3. Outputs a single file with prefixed columns for each condition:
   - {condition}_baseMean
   - {condition}_log2FoldChange
   - {condition}_lfcSE
   - {condition}_stat
   - {condition}_pvalue
   - {condition}_padj
 
Always uses pydeseq2 normalization (size factors).
Optionally supports batch correction via batch_column.
"""
 
import argparse
import time
import numpy as np
import pandas as pd
import anndata
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import warnings
import pickle
 
 
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Perform differential expression analysis for ALL conditions vs rest using PyDESeq2."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input h5ad file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output h5ad file.",
    )
    parser.add_argument(
        "--condition-column",
        required=True,
        help="Column in obs for conditions.",
    )
    parser.add_argument(
        "--batch-column",
        default=None,
        help="Column in obs for batch correction (optional).",
    )
    return parser.parse_args()
 
 
def prepare_counts_dataframe(adata):
    """
    Prepare counts DataFrame for PyDESeq2.
    PyDESeq2 expects samples as columns and features as rows.
    """
    if hasattr(adata.X, "toarray"):
        counts = adata.X.toarray()
    else:
        counts = adata.X
 
    counts_df = pd.DataFrame(counts.T, index=adata.var_names, columns=adata.obs_names)
    counts_df = counts_df.round().astype(int)
    return counts_df
 
 
def run_deseq2_for_condition(counts_df, metadata, target_condition, batch_column=None):
    """
    Run PyDESeq2 for one condition vs rest.
    Always uses pydeseq2 normalization (size factors).
 
    Parameters
    ----------
    counts_df : pd.DataFrame
        Counts matrix (features x samples)
    metadata : pd.DataFrame
        Sample metadata with 'condition' and optionally batch column
    target_condition : str
        Target condition to compare vs rest
    batch_column : str
        Column name for batch correction (optional)
 
    Returns
    -------
    pd.DataFrame
        DESeq2 results for this condition
    """
    # Create binary condition (target vs rest)
    meta_binary = metadata.copy()
    meta_binary["condition_binary"] = meta_binary["condition"].apply(
        lambda x: str(target_condition) if str(x) == str(target_condition) else "rest"
    )
    meta_binary["condition_binary"] = meta_binary["condition_binary"].astype(str)
 
    # Verify metadata index matches counts columns
    if not all(counts_df.columns == meta_binary.index):
        meta_binary = meta_binary.loc[counts_df.columns]
 
    # Create AnnData for DESeq2
    adata_for_deseq = anndata.AnnData(
        X=counts_df.T,  # Transpose: samples x features
        obs=meta_binary,
        var=pd.DataFrame(index=counts_df.index),
    )
 
    # Design formula with optional batch correction
    if batch_column:
        design_factors = [batch_column, "condition_binary"]
    else:
        design_factors = "condition_binary"
 
    # Create and run DESeq2
    dds = DeseqDataSet(
        adata=adata_for_deseq,
        design_factors=design_factors,
        ref_level=["condition_binary", "rest"],
        refit_cooks=True,
    )
 
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        dds.deseq2()  # Always use pydeseq2 normalization
 
    # Get statistics
    stat_res = DeseqStats(
        dds, contrast=["condition_binary", str(target_condition), "rest"], alpha=0.05
    )
    stat_res.summary()
 
    return stat_res.results_df, dds
 
 
def main():
    start_time = time.time()
    args = parse_arguments()
 
    print(f"Reading input file: {args.input}")
    adata = anndata.read_h5ad(args.input)
    print(f"Input shape: {adata.shape} (obs x var)")
 
    # Verify condition column exists
    if args.condition_column not in adata.obs.columns:
        raise ValueError(
            f"Condition column '{args.condition_column}' not found in obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
 
    # Get all unique conditions
    all_conditions = sorted(adata.obs[args.condition_column].unique())
    print(f"Found {len(all_conditions)} conditions: {all_conditions}")
 
    # Verify batch column if provided
    if args.batch_column:
        if args.batch_column not in adata.obs.columns:
            raise ValueError(
                f"Batch column '{args.batch_column}' not found in obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        print(f"Using batch correction with column: {args.batch_column}")
 
    # Prepare data
    counts_df = prepare_counts_dataframe(adata)
    metadata = pd.DataFrame(
        {"condition": adata.obs[args.condition_column].values}, index=adata.obs_names
    )
 
    if args.batch_column:
        metadata[args.batch_column] = adata.obs[args.batch_column].values
 
    # Run DESeq2 for each condition
    all_results = {}
    result_columns = ["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]
    normalized_counts = None
 
    for i, condition in enumerate(all_conditions):
        print(
            f"\n[{i+1}/{len(all_conditions)}] Running DESeq2 for {condition} vs rest..."
        )
 
        n_condition = (adata.obs[args.condition_column] == condition).sum()
        n_rest = (adata.obs[args.condition_column] != condition).sum()
        print(f"  Samples: {condition}={n_condition}, rest={n_rest}")
 
        try:
            results_df, dds = run_deseq2_for_condition(
                counts_df, metadata, condition, args.batch_column
            )
            all_results[condition] = results_df
 
            # Store normalized counts from first successful run
            if normalized_counts is None:
                normalized_counts = dds.layers["normed_counts"]
 
            n_significant = (results_df["padj"] < 0.05).sum()
            n_up = (
                (results_df["padj"] < 0.05) & (results_df["log2FoldChange"] > 0)
            ).sum()
            n_down = (
                (results_df["padj"] < 0.05) & (results_df["log2FoldChange"] < 0)
            ).sum()
            print(
                f"  Significant (padj < 0.05): {n_significant} (up: {n_up}, down: {n_down})"
            )
 
        except Exception as e:
            print(f"  ERROR for {condition}: {e}")
            # Create empty results with NaN
            all_results[condition] = pd.DataFrame(
                index=counts_df.index, columns=result_columns, data=np.nan
            )
        
        with open('all_results.pickle', 'wb') as file:
            pickle.dump(all_results, file)
        
        
 
    # Create output AnnData
    result_adata = anndata.AnnData(
        X=adata.X.copy(), obs=adata.obs.copy(), var=adata.var.copy()
    )
 
    # Add prefixed columns to var for each condition
    for condition, results_df in all_results.items():
        # Ensure results_df index matches var_names
        results_df = results_df.reindex(result_adata.var_names)
 
        for col in result_columns:
            col_name = f"{condition}_{col}"
            if col in results_df.columns:
                result_adata.var[col_name] = results_df[col].values
            else:
                result_adata.var[col_name] = np.nan
 
    # Add normalized counts if available
    if normalized_counts is not None:
        if normalized_counts.shape != (result_adata.n_obs, result_adata.n_vars):
            normalized_counts = normalized_counts.T
        result_adata.layers["normalized_counts"] = normalized_counts
 
    # Add metadata
    result_adata.uns["deseq2_params"] = {
        "design": (
            "condition" if not args.batch_column else f"{args.batch_column} + condition"
        ),
        "conditions": [str(c) for c in all_conditions],
        "condition_column": args.condition_column,
        "batch_column": args.batch_column,
        "n_features": int(result_adata.n_vars),
        "n_samples": int(result_adata.n_obs),
        "result_columns": [
            f"{c}_{col}" for c in all_conditions for col in result_columns
        ],
    }
 
    # Write output
    print(f"\nWriting results to: {args.output}")
    result_adata.write_h5ad(args.output)
 
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Conditions analyzed: {len(all_conditions)}")
    print(f"Features: {result_adata.n_vars}")
    print(f"Samples: {result_adata.n_obs}")
    print(f"Output columns per condition: {result_columns}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print(f"{'='*60}")
 
 
if __name__ == "__main__":
    main()
 
 