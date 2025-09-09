import pandas as pd
import numpy as np

from functools import reduce

def load_and_merge_reports(paths, columns_to_rename=None):
    """
    Load multiple JSON report files, rename their columns based on report_source,
    and merge them into a single DataFrame.

    Parameters:
    - paths: list of str, paths to JSON files
    - columns_to_rename: list of column names to rename (default: ['prediction', 'confidence'])

    Returns:
    - Merged pandas DataFrame
    """
    
    if columns_to_rename is None:
        columns_to_rename = ['prediction', 'confidence']
    
    def rename_columns_by_report(df):
        # Determine the suffix based on report_source
        suffix_map = {
            "gt": "_usinggt",
            "llm": "_usingllm"
        }
        suffix = "_usingnoreport"  # default

        for key, val in suffix_map.items():
            if (df["report_source"] == key).any():
                suffix = val
                break

        # Rename specified columns
        renamed_columns = {
            col: f"{col}{suffix}" for col in columns_to_rename if col in df.columns
        }

        # Drop extra columns if they exist
        df = df.drop(columns=[c for c in ['report_source', 'choices'] if c in df.columns])

        return df.rename(columns=renamed_columns)

    dataframes = [rename_columns_by_report(pd.read_json(path)) for path in paths]

    merge_keys = ["report_id", "question", "question_source", "answer"]
    merged = reduce(lambda left, right: left.merge(right, on=merge_keys, how="left"), dataframes)

    return merged


def calculate_agreement(df, col_gt='prediction_usinggt', col_gen='prediction_usingnoreport'):
    """
    Calculate agreement and disagreement percentages at the dataset level,
    separately for 'gt' and 'llm' question sources.
    
    Args:
        df: pandas DataFrame containing predictions
        col_gt: column name for predictions using GT
        col_gen: column name for predictions using generated/LLM
        
    Returns:
        Dictionary containing agreement and disagreement percentages overall,
        and separately for GT and LLM question sources.
    """
    try:
        if col_gt not in df.columns or col_gen not in df.columns:
            raise ValueError(f"Columns '{col_gt}' or '{col_gen}' not found in the DataFrame.")
        
        def _compute_agreement(sub_df):
            total = len(sub_df)
            if total == 0:
                return {
                    'total_questions': 0,
                    'agreement_count': 0,
                    'disagreement_count': 0,
                    'agreement_percentage': None,
                    'disagreement_percentage': None
                }
            agreements = sub_df[col_gt] == sub_df[col_gen]
            agreement_count = agreements.sum()
            disagreement_count = total - agreement_count
            return {
                'total_questions': total,
                'agreement_count': int(agreement_count),
                'disagreement_count': int(disagreement_count),
                'agreement_percentage': round((agreement_count / total) * 100, 2),
                'disagreement_percentage': round((disagreement_count / total) * 100, 2)
            }
        
        # Overall agreement
        overall = _compute_agreement(df)
        
        # Agreement for GT questions
        gt_df = df[df['question_source'] == 'gt']
        agreement_gt = _compute_agreement(gt_df)
        
        # Agreement for LLM questions
        llm_df = df[df['question_source'] == 'llm']
        agreement_llm = _compute_agreement(llm_df)

        result = {
            'overall': overall,
            'agreement_gt': agreement_gt,
            'agreement_llm': agreement_llm
        }

        # Convert each section into a DataFrame for nicer display
        overall_df = pd.DataFrame([result['overall']])
        gt_df = pd.DataFrame([result['agreement_gt']])
        llm_df = pd.DataFrame([result['agreement_llm']])
        
        print("=== Overall Agreement ===")
        print(overall_df, "\n")
        
        print("=== GT Questions (Accuracy) ===")
        print(gt_df, "\n")
        
        print("=== LLM Questions (Recall Proxy) ===")
        print(llm_df, "\n")
        
        return result
        
    except Exception as e:
        print(f"Error calculating dataset agreement: {e}")
        return None
    

def plot_report_level_agreement(df, output_dir: str):
    """
    Plot report-level agreement statistics.
    
    Args:
        csv_file: Path to the CSV file containing predictions
        output_dir: Directory to save the plot
        reference: Reference type ('gt' or 'gen')
    """
    try:
        report_agreements = []
        
        # Calculate agreement percentage for each report
        print(len(df['report_id'].unique()))
        for report_id in df['report_id'].unique():
            report_df = df[df['report_id'] == report_id]
            total = len(report_df)
            matches = len(report_df[report_df['prediction_usinggt'] == 
                                  report_df['prediction_usingllm']])
            agreement_pct = (matches / total) * 100 if total > 0 else 0
            report_agreements.append(agreement_pct)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.hist(report_agreements, bins=20, edgecolor='black')
        plt.xlabel('Agreement Percentage', fontsize=24, fontweight='bold', labelpad=15)
        plt.ylabel('Number of Reports', fontsize=24, fontweight='bold', labelpad=15)
        # plt.title(f'Distribution of GT-GEN Agreement Across Reports\n({reference.upper()} Reference)', fontsize=24)
        plt.xticks(fontsize=22, weight='bold')
        plt.yticks(fontsize=22, weight='bold')
        
        # Add mean and std dev lines
        mean_agreement = np.mean(report_agreements)
        print(mean_agreement)
        std_agreement = np.std(report_agreements)
        plt.axvline(mean_agreement, color='r', linestyle='dashed', linewidth=2, 
                   label=f'Mean: {mean_agreement:.1f}%')
        plt.axvline(mean_agreement + std_agreement, color='g', linestyle=':', linewidth=2,
                   label=f'SD: {std_agreement:.1f}%')
        plt.axvline(mean_agreement - std_agreement, color='g', linestyle=':', linewidth=2)
        
        plt.legend(fontsize=24, prop={'weight': 'bold', 'size': 16})
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = os.path.join(output_dir, f'mcq_eval_report_level_agreement_hist.png')
        plt.savefig(plot_file, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # Save report-level statistics
        report_stats = pd.DataFrame({
            'Report_ID': df['report_id'].unique(),
            'Agreement_Percentage': report_agreements
        })
        stats_file = os.path.join(output_dir, f'mcq_eval_report_level_stats.csv')
        report_stats.to_csv(stats_file, index=False)

        aggregated_stats_file = os.path.join(output_dir, f'mcq_eval_report_level_stats_aggregated.csv')
        aggregated_stats = pd.DataFrame({
            'Mean_Agreement': [mean_agreement],
            'Std_Deviation': [std_agreement]
        })
        aggregated_stats.to_csv(aggregated_stats_file, index=False)
        
        print(f"\nReport-level statistics:")
        print(f"Mean agreement: {mean_agreement:.1f}%")
        print(f"Standard deviation: {std_agreement:.1f}%")
        print(f"Plot saved to {plot_file}")
        print(f"Report-level statistics saved to {stats_file}")
        
    except Exception as e:
        print(f"Error plotting report-level agreement: {e}")