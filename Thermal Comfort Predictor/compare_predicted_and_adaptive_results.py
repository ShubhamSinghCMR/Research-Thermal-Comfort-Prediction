import pandas as pd

def compare_results(original_results, adaptive_results, output_file="comparison_results.csv"):
    # Load CSVs
    predicted_tsv_df = pd.read_csv(original_results)
    adaptive_tsv_df = pd.read_csv(adaptive_results)

    # Merge on Environment
    comparison = pd.merge(predicted_tsv_df, adaptive_tsv_df, on="Environment", suffixes=("_predicted_tsv", "_adaptive_tsv"))

    # Metrics to compare
    metrics = ["RMSE", "MAE", "R2", "Accuracy", "Residual_STD", "MBE"]

    # Build a new DataFrame with columns grouped (Predicted, Adaptive, Difference)
    formatted_data = []
    for _, row in comparison.iterrows():
        row_data = {"Environment": row["Environment"]}
        for metric in metrics:
            # For the first merge, there won't be a suffix for predicted_tsv columns
            old_val = row[f"{metric}_predicted_tsv"] if f"{metric}_predicted_tsv" in row else row[metric]
            new_val = row[f"{metric}_adaptive_tsv"]
            diff_val = new_val - old_val
            row_data[f"**{metric} (Predicted_TSV)**"] = round(old_val, 4)
            row_data[f"**{metric} (Adaptive_TSV)**"] = round(new_val, 4)
            row_data[f"**{metric} (Difference)**"] = round(diff_val, 4)
        formatted_data.append(row_data)

    formatted_df = pd.DataFrame(formatted_data)

    # Save as CSV
    formatted_df.to_csv(output_file, index=False)
    print(f"[📂 Saved] Comparison results saved to {output_file}")

    # Print in console
    for _, row in formatted_df.iterrows():
        print(f"\n=== {row['Environment']} ===")
        for metric in metrics:
            print(f"{metric}: Predicted_TSV={row[f'**{metric} (Predicted_TSV)**']} | "
                  f"Adaptive_TSV={row[f'**{metric} (Adaptive_TSV)**']} | "
                  f"**Diff={row[f'**{metric} (Difference)**']}**")

if __name__ == "__main__":
    original_results = "output/predicted_tsv_results.csv"
    adaptive_results = "adaptive_tsv_pipeline/output/adaptive_tsv_results.csv"
    compare_results(original_results, adaptive_results)
