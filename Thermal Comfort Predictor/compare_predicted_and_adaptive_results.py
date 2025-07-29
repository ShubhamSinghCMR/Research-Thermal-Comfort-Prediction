import pandas as pd

def compare_results(original_results, adaptive_results, output_file="comparison_results.csv"):
    # Load CSVs
    predicted_tsv_df = pd.read_csv(original_results)
    adaptive_tsv_df = pd.read_csv(adaptive_results)

    # Merge on Environment
    comparison = pd.merge(predicted_tsv_df, adaptive_tsv_df, on="Environment", suffixes=("_predicted_tsv", "_adaptive_tsv"))

    # Metrics to compare
    metrics = ["RMSE", "MAE", "R2", "Accuracy", "Residual_STD", "MBE"]
    lower_better = ["RMSE", "MAE", "Residual_STD", "MBE"]

    formatted_data = []
    total_adaptive_better = 0
    total_predicted_better = 0

    for _, row in comparison.iterrows():
        row_data = {"Environment": row["Environment"]}
        adaptive_better_count = 0
        predicted_better_count = 0

        for metric in metrics:
            old_val = row[f"{metric}_predicted_tsv"]
            new_val = row[f"{metric}_adaptive_tsv"]
            diff_val = new_val - old_val

            # Decide which is better
            if metric in lower_better:
                if diff_val < 0:
                    status = "Adaptive_TSV is Better ✅"
                    adaptive_better_count += 1
                elif diff_val > 0:
                    status = "Predicted_TSV is Better ✅"
                    predicted_better_count += 1
                else:
                    status = "No change"
            else:  # higher is better
                if diff_val > 0:
                    status = "Adaptive_TSV is Better ✅"
                    adaptive_better_count += 1
                elif diff_val < 0:
                    status = "Predicted_TSV is Better ✅"
                    predicted_better_count += 1
                else:
                    status = "No change"

            row_data[f"**{metric} (Predicted_TSV)**"] = round(old_val, 4)
            row_data[f"**{metric} (Adaptive_TSV)**"] = round(new_val, 4)
            row_data[f"**{metric} (Difference)**"] = round(diff_val, 4)
            row_data[f"**{metric} (Better)**"] = status

        # Overall winner for this environment
        if adaptive_better_count > predicted_better_count:
            row_data["**Overall Winner (Environment)**"] = "Adaptive_TSV"
            total_adaptive_better += 1
        elif predicted_better_count > adaptive_better_count:
            row_data["**Overall Winner (Environment)**"] = "Predicted_TSV"
            total_predicted_better += 1
        else:
            row_data["**Overall Winner (Environment)**"] = "Tie"

        formatted_data.append(row_data)

    formatted_df = pd.DataFrame(formatted_data)

    # Save CSV
    formatted_df.to_csv(output_file, index=False)
    

    # Print summary in console
    for _, row in formatted_df.iterrows():
        print(f"\n=== {row['Environment']} ===")
        for metric in metrics:
            print(f"{metric}: Predicted_TSV={row[f'**{metric} (Predicted_TSV)**']} | "
                  f"Adaptive_TSV={row[f'**{metric} (Adaptive_TSV)**']} | "
                  f"Diff={row[f'**{metric} (Difference)**']} → {row[f'**{metric} (Better)**']}")
        print(f"Overall (Environment): {row['**Overall Winner (Environment)**']}")
    
    print(f"\nComparison results saved to {output_file}\n")

if __name__ == "__main__":
    original_results = "output/predicted_tsv_results.csv"
    adaptive_results = "adaptive_tsv_pipeline/output/adaptive_tsv_results.csv"
    compare_results(original_results, adaptive_results)
