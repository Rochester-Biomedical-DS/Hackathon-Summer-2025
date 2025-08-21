# catboost_runner.py
from catboost import CatBoostRegressor, Pool
import argparse
import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path


def read_train_wide(train_csv: str) -> pd.DataFrame:
    """
    Reads the wide training matrix:
      rows = genes (e.g., g0001, g0002, ...)
      columns = perturbations (e.g., g0495+ctrl, g0261+g0..., ...)
      values = expression (float)
    Ensures there's a 'gene' column.
    """
    df = pd.read_csv(train_csv)
    # If first column is the gene id but not named 'gene', rename it.
    first_col = df.columns[0]
    if first_col.lower() != "gene":
        df = df.rename(columns={first_col: "gene"})
    return df


def melt_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Converts wide matrix to long tidy format with columns:
      gene, perturbation, expression
    Drops non-numeric / NaN expressions.
    """
    long_df = df_wide.melt(
        id_vars=["gene"], var_name="perturbation", value_name="expression")
    # Coerce to numeric in case CSV had strings; drop NaNs
    long_df["expression"] = pd.to_numeric(
        long_df["expression"], errors="coerce")
    long_df = long_df.dropna(subset=["expression"]).reset_index(drop=True)
    return long_df


def train_catboost(long_df: pd.DataFrame,
                   iterations: int = 1200,
                   depth: int = 6,
                   learning_rate: float = 0.05,
                   random_seed: int = 42,
                   verbose: int = 200) -> CatBoostRegressor:
    """
    Trains a CatBoostRegressor using 'gene' and 'perturbation' as categorical features.
    """
    X = long_df[["gene", "perturbation"]]
    y = long_df["expression"].values
    train_pool = Pool(X, y, cat_features=["gene", "perturbation"])

    model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="RMSE",
        random_seed=random_seed,
        verbose=verbose
    )
    model.fit(train_pool)
    return model


def read_test_perturbations(test_csv: str) -> pd.Series:
    """
    Reads test_set.csv which has a single column called 'perturbation'.
    If header is missing or named differently, coerces to that name.
    """
    tdf = pd.read_csv(test_csv)
    if "perturbation" not in tdf.columns:
        # assume first column holds perturbation strings
        tdf = tdf.rename(columns={tdf.columns[0]: "perturbation"})
    # Clean up whitespace just in case
    tdf["perturbation"] = tdf["perturbation"].astype(str).str.strip()
    return tdf["perturbation"]


def cartesian_genes_perturbations(genes: pd.Series, perturbations: pd.Series) -> pd.DataFrame:
    """
    Builds all gene × perturbation pairs for inference.
    """
    # Efficient cartesian product using repeat/tile
    g = np.repeat(genes.values, len(perturbations))
    p = np.tile(perturbations.values, len(genes))
    df = pd.DataFrame({"gene": g, "perturbation": p})
    return df


def write_predictions_csv(output_csv: str, genes: pd.Series, perturbations: pd.Series, preds: np.ndarray):
    """
    Writes predictions in the expected format:
      gene,perturbation,expression
    Maintaining the exact header and style used in your pseudoinverse pipeline.
    """
    outdir = os.path.dirname(output_csv)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # preds are aligned with the cartesian order we created
    result = pd.DataFrame({
        "gene": np.repeat(genes.values, len(perturbations)),
        "perturbation": np.tile(perturbations.values, len(genes)),
        "expression": preds
    })
    result.to_csv(output_csv, index=False, columns=[
                  "gene", "perturbation", "expression"])


def main():
    parser = argparse.ArgumentParser(
        description="Train CatBoost on (gene × perturbation) matrix and predict for test perturbations.")
    parser.add_argument("--train", default="data/train_set.csv")
    parser.add_argument("--test", default="data/test_set.csv")
    parser.add_argument("--out", default="prediction/prediction.csv")
    # Optional CatBoost args
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=200)
    args = parser.parse_args()

    # 1) Load & reshape training data
    df_wide = read_train_wide(args.train)
    if "gene" not in df_wide.columns:
        raise ValueError(
            "Could not find a 'gene' column in the training data after reading. Please check the CSV.")

    genes = df_wide["gene"].astype(str)
    long_df = melt_to_long(df_wide)

    # 2) Train CatBoost on (gene, perturbation) categorical features
    print(
        f"Training CatBoost on {len(long_df):,} (gene, perturbation) samples...")
    model = train_catboost(
        long_df,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.lr,
        random_seed=args.seed,
        verbose=args.verbose
    )

    # 3) Prepare inference pairs: all genes × each test perturbation
    test_perturbs = read_test_perturbations(args.test)
    pairs_df = cartesian_genes_perturbations(genes, test_perturbs)

    # 4) Predict
    infer_pool = Pool(pairs_df[["gene", "perturbation"]],
                      cat_features=["gene", "perturbation"])
    preds = model.predict(infer_pool).astype(float)

    # 5) Write predictions in the SAME format as pseudoinverse
    write_predictions_csv(args.out, genes, test_perturbs, preds)
    print(f"Saved predictions → {args.out}")
    print("PROCESS FINISHED")


if __name__ == "__main__":
    main()
