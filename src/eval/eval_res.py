import pandas as pd
import os
from plotnine import (
    ggplot, aes, geom_histogram, geom_vline, geom_boxplot, geom_jitter,
    geom_point, geom_abline, geom_text,
    labs, theme_bw, theme, element_text, scale_fill_manual, scale_y_continuous, scale_x_continuous,
    scale_color_manual
)
from functools import reduce


def load_and_merge_reports(
        paths, columns_to_rename=None
        ):
    """
    Load multiple JSON report files, rename specified columns using the value 
    in 'report_source', and merge them into a single DataFrame.

    Parameters:
    - paths: list of str, paths to JSON files
    - columns_to_rename: list of column names to rename 
                         (default: ['prediction', 'confidence'])

    Returns:
    - Merged pandas DataFrame
    """
    
    if columns_to_rename is None:
        columns_to_rename = ['prediction', 'confidence']

    def rename_columns_by_report(df):
        if "report_source" not in df.columns:
            raise ValueError("Missing 'report_source' column in one of the reports")

        # Get the unique report_source value (assume consistent within file)
        source_val = df["report_source"].iloc[0]

        # Rename specified columns with suffix = report_source value
        renamed_columns = {
            col: f"{col}_{source_val}" for col in columns_to_rename if col in df.columns
        }

        # Drop metadata columns if they exist
        df = df.drop(columns=[c for c in ['report_source', 'choices'] if c in df.columns])

        return df.rename(columns=renamed_columns)

    # Process all reports
    dataframes = [rename_columns_by_report(pd.read_json(path)) for path in paths]

    # Merge on common keys
    merge_keys = ["task_id", "question", "question_source", "answer"]
    merged = reduce(lambda left, right: left.merge(right, on=merge_keys, how="left"), dataframes)

    return merged


def calculate_agreement(df):
    """
    Calculate agreement/disagreement 
    at dataset and subset levels. Meaning per question.
    Automatically detect prediction columns and compare the first two found.
    Args:
        df: pandas DataFrame containing at least two columns with 'prediction' in their names.

    Returns:
        Dictionary containing agreement/disagreement percentages overall,
        and separately for GT and generated report questions.
    """
    try:
        # --- Helpers ---
        def get_suffix(col_name: str) -> str:
            """Extract suffix after 'prediction_' (e.g., 'gt', 'llm', 'noreport')."""
            return col_name.split("prediction_")[-1]

        def _compute_agreement(sub_df, col_a, col_b):
            """Compute agreement stats between two prediction columns."""
            total = len(sub_df)
            if total == 0:
                return {
                    'total_questions': 0,
                    'agreement_count': 0,
                    'disagreement_count': 0,
                    'agreement_percentage': None,
                    'disagreement_percentage': None
                }
            agreements = sub_df[col_a] == sub_df[col_b]
            agreement_count = int(agreements.sum())
            disagreement_count = total - agreement_count
            return {
                'total_questions': total,
                'agreement_count': agreement_count,
                'disagreement_count': disagreement_count,
                'agreement_percentage': round((agreement_count / total) * 100, 2),
                'disagreement_percentage': round((disagreement_count / total) * 100, 2)
            }

        # --- Find prediction columns ---
        pred_cols = [col for col in df.columns if "prediction" in col.lower()]
        if len(pred_cols) < 2:
            raise ValueError(f"Expected at least two 'prediction' columns, found: {pred_cols}")

        col1, col2 = pred_cols[:2]
        suffix1, suffix2 = get_suffix(col1), get_suffix(col2)

        # --- Assign GT vs generated report ---
        if suffix1 in ("gt", "groundtruth"):
            gt_col, gen_col = col1, col2
            gt_suffix, gen_suffix = suffix1, suffix2
        else:
            gt_col, gen_col = col2, col1
            gt_suffix, gen_suffix = suffix2, suffix1

        # --- Compute agreements ---
        overall = _compute_agreement(df, col1, col2)
        agreement_gt = _compute_agreement(df[df['question_source'] == gt_suffix], gt_col, gen_col)
        agreement_gen = _compute_agreement(df[df['question_source'] == gen_suffix], gt_col, gen_col)

        result = {
            'columns_compared': (col1, col2),
            'overall': overall,
            f'agreement_{gt_suffix}': agreement_gt,
            f'agreement_{gen_suffix}': agreement_gen
        }

        # --- Pretty printing ---
        print(f"=== Comparing: {col1} vs {col2} ===\n")
        print("=== Overall Agreement on Total Questions ===\n")
        print(pd.DataFrame([overall]), "\n")
        
        print(f"=== Agreement on Questions Based on Content from {gt_suffix.upper()} Report ===\n")
        print(pd.DataFrame([agreement_gt]), "\n")
        
        print(f"=== Agreement on Questions Generated Based on Content from {gen_suffix.upper()} Report ===\n")
        print(pd.DataFrame([agreement_gen]), "\n")
        
        return result

    except Exception as e:
        print(f"Error calculating dataset agreement: {e}")
        return None


def agreement_x_analysis(df, output_dir="../output/"):
    """
    Generate publication-quality agreement plots using plotnine (ggplot style).
    
    Args:
        df: pandas DataFrame with predictions and task_id
        output_dir: directory to save plots
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # --- Detect prediction columns ---
        pred_cols = [col for col in df.columns if "prediction" in col.lower()]
        if len(pred_cols) < 2:
            raise ValueError(f"Expected at least two 'prediction' columns, found: {pred_cols}")
        col1, col2 = pred_cols[:2]

        # --- Compute overall and disaggregated agreement ---
        report_agreements = []
        disaggregated = []

        for task_id in df['task_id'].unique():
            report_df = df[df['task_id'] == task_id]

            # Overall agreement
            total_quest = len(report_df)
            matches = (report_df[col1] == report_df[col2]).sum()
            overall_pct = (matches / total_quest) * 100 if total_quest > 0 else 0
            report_agreements.append({"Task_ID": task_id, "Agreement_Percentage": overall_pct})

            # Disaggregated by question_source
            for source, sub_df in report_df.groupby("question_source"):
                total_src = len(sub_df)
                matches_src = (sub_df[col1] == sub_df[col2]).sum()
                pct_src = (matches_src / total_src) * 100 if total_src > 0 else 0
                disaggregated.append({
                    "Task_ID": task_id,
                    "Question_Source": source,
                    "Agreement_Percentage": pct_src
                })

        overall_df = pd.DataFrame(report_agreements)
        disagg_df = pd.DataFrame(disaggregated)

        # --- Plot 1: Overall histogram ---
        mean_overall = overall_df["Agreement_Percentage"].mean()
        std_overall = overall_df["Agreement_Percentage"].std()

        vlines_df = pd.DataFrame({
            "value": [mean_overall, mean_overall+std_overall, mean_overall-std_overall],
            "type": ["Mean", "Mean ± 1 SD", "Mean ± 1 SD"]  # combine SD lines under one label
        })

        p1 = (
            ggplot(overall_df, aes(x="Agreement_Percentage")) +
            geom_histogram(binwidth=5, fill="#999999", color="black", alpha=0.8) +
            geom_vline(aes(xintercept="value", color="type"), data=vlines_df, linetype="dashed", size=0.6) +
            scale_color_manual(values={"Mean": "blue", "Mean ± 1 SD": "green"}) +
            labs(
                title="Overall Task-Level Agreement",
                x="Agreement Percentage",
                y="Number of Tasks",
                color="Statistics"
            ) +
            theme_bw() +
            theme(
                figure_size=(8,6),
                plot_title=element_text(size=16, weight="bold"),
                axis_title=element_text(size=14, weight="bold"),
                axis_text=element_text(size=12)
            )
        )
        p1.save(os.path.join(output_dir, "agreement_hist_overall_gg.png"), dpi=600)

        # --- Plot 2: Disaggregated histogram by question_source ---
        p2 = (
            ggplot(disagg_df, aes(x="Agreement_Percentage", fill="Question_Source")) +
            geom_histogram(binwidth=5, position="identity", alpha=0.6, color="black") +
            scale_fill_manual(values=["#4C72B0", "#55A868"]) +
            labs(title="Agreement by Question Source", x="Agreement Percentage", y="Number of Reports") +
            theme_bw() +
            theme(
                figure_size=(8,6),
                plot_title=element_text(size=16, weight="bold"),
                axis_title=element_text(size=14, weight="bold"),
                axis_text=element_text(size=12),
                legend_title=element_text(size=12, weight="bold"),
                legend_text=element_text(size=12)
            )
        )
        p2.save(os.path.join(output_dir, "agreement_hist_by_source_gg.png"), dpi=600)

        # --- Plot 3: Boxplot by question_source ---
        p3 = (
            ggplot(disagg_df, aes(x="Question_Source", y="Agreement_Percentage", fill="Question_Source")) +
            geom_boxplot(alpha=0.6) +
            geom_jitter(width=0.2, alpha=0.5, size=2, color="black") +
            scale_fill_manual(values=["#4C72B0", "#55A868"]) +
            labs(title="Agreement Distribution by Question Source", x="Question Source", y="Agreement Percentage") +
            theme_bw() +
            theme(
                figure_size=(8,6),
                plot_title=element_text(size=16, weight="bold"),
                axis_title=element_text(size=14, weight="bold"),
                axis_text=element_text(size=12),
                legend_position="none"
            )
        )
        p3.save(os.path.join(output_dir, "agreement_boxplot_by_source_gg.png"), dpi=600)


        # --- Plot 4: Task-level GT vs LLM scatter with annotations using question_source column ---
        # Identify the unique sources
        sources = disagg_df['Question_Source'].unique()
        if len(sources) < 2:
            raise ValueError(f"Expected at least two question sources, found: {sources}")
        gt_source = [s for s in sources if 'gt' in s.lower() or 'groundtruth' in s.lower()][0]
        llm_source = [s for s in sources if s != gt_source][0]

        # Pivot data using these dynamic sources
        pivot_df = disagg_df.pivot(index='Task_ID', columns='Question_Source', values='Agreement_Percentage').reset_index()
        pivot_df.columns.name = None

        # Make sure the pivoted columns exist
        if gt_source not in pivot_df.columns or llm_source not in pivot_df.columns:
            raise ValueError(f"Pivoted columns not found: {gt_source}, {llm_source}")

        p4 = (
            ggplot(pivot_df, aes(x=gt_source, y=llm_source, label='Task_ID')) +
            geom_point(size=3, color="#4C72B0", alpha=0.7) +
            geom_text(aes(x=gt_source, y=llm_source),
                    nudge_x=1.5, nudge_y=1.5, size=8, ha='left') +
            geom_abline(slope=1, intercept=0, linetype='dashed', color='red') +
            scale_x_continuous(limits=(0, 100)) +
            scale_y_continuous(limits=(0, 100)) +
            labs(
                title=(
                    f"Task-Level Answer Agreement (%)\n\n"
                    f"Questions from {gt_source.upper()}\n"
                    f"vs\n"
                    f"Questions from {llm_source.upper()}\n"
                ),
                x=f"Agreement on Questions from {gt_source.upper()}",
                y=f"Agreement on Questions from {llm_source.upper()}"
            ) +
            theme_bw() +
            theme(
                figure_size=(8,8),
                plot_title=element_text(size=14, weight='bold', hjust=0.5),
                axis_title=element_text(size=14, weight='bold'),
                axis_text=element_text(size=12)
            )
        )
        p4.save(os.path.join(output_dir, "task_level_agreement_gt_vs_llm_annotated.png"), dpi=600)

        print(f"GGPlot-style plots saved in {output_dir}")

        # --- Save statistics ---
        overall_df.to_csv(os.path.join(output_dir, "agreement_report_level.csv"), index=False)
        disagg_df.to_csv(os.path.join(output_dir, "agreement_report_level_by_source.csv"), index=False)

        aggregated_stats = pd.DataFrame({
            "Mean_Agreement": [mean_overall],
            "Std_Deviation": [std_overall]
        })
        aggregated_stats.to_csv(os.path.join(output_dir, "agreement_report_level_aggregated.csv"), index=False)

        print(f"Mean agreement: {mean_overall:.1f}%, Std Dev: {std_overall:.1f}%")

    except Exception as e:
        print(f"Error generating ggplot-style agreement plots: {e}")

