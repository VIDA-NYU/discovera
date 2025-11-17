import pandas as pd
import os
import json
import torch
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from plotnine import (
    ggplot, aes, geom_histogram, geom_vline, geom_boxplot, geom_jitter,
    geom_point, geom_abline, geom_text,
    labs, theme_bw, theme, element_text, scale_fill_manual, scale_y_continuous, scale_x_continuous,
    scale_color_manual
)
from functools import reduce
from pathlib import Path

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

def calculate_agreement(df, output_dir="../../output/"):
    """
    Calculate agreement/disagreement per question and per dataset subset.
    Automatically detect prediction columns and compare the first two found.

    Args:
        df: pandas DataFrame containing at least two columns with 'prediction' in their names.
        output_dir: directory to save results.

    Returns:
        Tuple: (summary_dict, summary_df, detailed_df)
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

        results_dict = {
            'columns_compared': (col1, col2),
            'overall': overall,
            f'agreement_{gt_suffix}': agreement_gt,
            f'agreement_{gen_suffix}': agreement_gen
        }

        # --- Create summary DataFrame ---
        results_df = pd.DataFrame([
            {"level": "overall", **overall},
            {"level": f"{gt_suffix}_questions", **agreement_gt},
            {"level": f"{gen_suffix}_questions", **agreement_gen}
        ])

        # --- Create detailed per-question DataFrame ---
        detailed_df = df.copy()
        detailed_df["agreement"] = (df[col1] == df[col2]).astype(int)
        detailed_df["agree_label"] = detailed_df["agreement"].map({1: "agree", 0: "disagree"})
        detailed_df = detailed_df[
            ["task_id", "question", "question_source", col1, col2, "agreement", "agree_label"]
        ].sort_values(["task_id", "question_source", "question"])

        # --- Save both CSVs ---
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            results_df.to_csv(os.path.join(output_dir, "agreement_summary.csv"), index=False)
            detailed_df.to_csv(os.path.join(output_dir, "agreement_per_question.csv"), index=False)
            print(f"Saved summary and per-question files to {output_dir}")

        # --- Pretty printing ---
        print(f"\n=== Comparing: {col1} vs {col2} ===\n")
        print(results_df.to_string(index=False))

        return results_dict, results_df, detailed_df

    except Exception as e:
        print(f"Error calculating dataset agreement: {e}")
        return None, None, None


# def calculate_agreement(df, output_dir="../../output/"):
#     """
#     Calculate agreement/disagreement 
#     at dataset and subset levels. Meaning per question.
#     Automatically detect prediction columns and compare the first two found.
#     Args:
#         df: pandas DataFrame containing at least two columns with 'prediction' in their names.

#     Returns:
#         Dictionary containing agreement/disagreement percentages overall,
#         and separately for GT and generated report questions.
#     """
#     try:
#         # --- Helpers ---
#         def get_suffix(col_name: str) -> str:
#             """Extract suffix after 'prediction_' (e.g., 'gt', 'llm', 'noreport')."""
#             return col_name.split("prediction_")[-1]

#         def _compute_agreement(sub_df, col_a, col_b):
#             """Compute agreement stats between two prediction columns."""
#             total = len(sub_df)
#             if total == 0:
#                 return {
#                     'total_questions': 0,
#                     'agreement_count': 0,
#                     'disagreement_count': 0,
#                     'agreement_percentage': None,
#                     'disagreement_percentage': None
#                 }
#             agreements = sub_df[col_a] == sub_df[col_b]
#             agreement_count = int(agreements.sum())
#             disagreement_count = total - agreement_count
#             return {
#                 'total_questions': total,
#                 'agreement_count': agreement_count,
#                 'disagreement_count': disagreement_count,
#                 'agreement_percentage': round((agreement_count / total) * 100, 2),
#                 'disagreement_percentage': round((disagreement_count / total) * 100, 2)
#             }

#         # --- Find prediction columns ---
#         pred_cols = [col for col in df.columns if "prediction" in col.lower()]
#         if len(pred_cols) < 2:
#             raise ValueError(f"Expected at least two 'prediction' columns, found: {pred_cols}")

#         col1, col2 = pred_cols[:2]
#         suffix1, suffix2 = get_suffix(col1), get_suffix(col2)

#         # --- Assign GT vs generated report ---
#         if suffix1 in ("gt", "groundtruth"):
#             gt_col, gen_col = col1, col2
#             gt_suffix, gen_suffix = suffix1, suffix2
#         else:
#             gt_col, gen_col = col2, col1
#             gt_suffix, gen_suffix = suffix2, suffix1

#         # --- Compute agreements ---
#         overall = _compute_agreement(df, col1, col2)
#         agreement_gt = _compute_agreement(df[df['question_source'] == gt_suffix], gt_col, gen_col)
#         agreement_gen = _compute_agreement(df[df['question_source'] == gen_suffix], gt_col, gen_col)

#         results_dict = {
#             'columns_compared': (col1, col2),
#             'overall': overall,
#             f'agreement_{gt_suffix}': agreement_gt,
#             f'agreement_{gen_suffix}': agreement_gen
#         }

#         # --- Create tidy DataFrame ---
#         results_df = pd.DataFrame([
#             {"level": "overall", **overall},
#             {"level": f"{gt_suffix}_questions", **agreement_gt},
#             {"level": f"{gen_suffix}_questions", **agreement_gen}
#         ])

#         # --- Save CSV if requested ---
#         if output_dir:
#             results_df.to_csv(os.path.join(output_dir, "accuracy_precision_recall.csv"), index=False)
#             print(f"Results saved to {output_dir}")

#         # --- Pretty printing ---
#         print(f"=== Comparing: {col1} vs {col2} ===\n")
#         print(results_df.to_string(index=False))

#         return results_dict, results_df

#     except Exception as e:
#         print(f"Error calculating dataset agreement: {e}")
#         return None, None

from plotnine import (
    ggplot, aes, geom_tile,
    scale_fill_manual, labs, theme_bw, theme, element_text
)

def plot_task_heatmaps_by_source(df, output_dir="../../output/"):
    """
    Generate one heatmap per question_source, showing agreement per task and question.
    
    Each plot:
      - X-axis: question
      - Y-axis: task_id
      - Color: agreement (1 = agree, 0 = disagree)
    
    Saves one PNG file per question_source.
    """
    # Ensure needed columns
    required_cols = {"task_id", "question", "question_source", "agreement"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

    os.makedirs(output_dir, exist_ok=True)

    # Iterate over question sources (e.g., groundtruth, llm)
    for source, subdf in df.groupby("question_source"):
        subdf = subdf.copy()
        subdf["task_id"] = subdf["task_id"].astype(str)

        p = (
            ggplot(subdf, aes(x="question", y="task_id", fill="factor(agreement)"))
            + geom_tile(color="white")
            + scale_fill_manual(
                values={"1": "#43aa8b", "0": "#f94144"},
                name="Agreement",
                labels=["Disagree", "Agree"]
            )
            + labs(
                title=f"Agreement Heatmap — Question Source: {source}",
                x="Question",
                y="Task ID"
            )
            + theme_bw()
            + theme(
                axis_text_x=element_text(rotation=90, size=6),
                axis_text_y=element_text(size=7),
                figure_size=(10, 6),
                legend_title=element_text(size=9),
                legend_text=element_text(size=8),
                plot_title=element_text(size=12, weight="bold")
            )
        )

        plot_path = os.path.join(output_dir, f"agreement_heatmap_{source}.png")
        p.save(plot_path, dpi=300)
        print(f"Saved heatmap for '{source}' to {plot_path}")

    print("\n✅ All heatmaps saved.")



def agreement_x_analysis(df, output_dir="../../output/"):
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
        print(mean_overall)
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
            scale_x_continuous(limits=(0,100)) +
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
        # find unique sources
        sources = disagg_df["Question_Source"].unique()
        # Make sure groundtruth is first
        ordered_sources = sorted(sources, key=lambda s: 0 if "groundtruth" in s.lower() else 1)

        # Build color map accordingly
        color_map = {
            s: ("#4C72B0" if "groundtruth" in s.lower() else "#55A868")
            for s in ordered_sources
        }
        p2 = (
            ggplot(disagg_df, aes(x="Agreement_Percentage", fill="Question_Source")) +
            geom_histogram(binwidth=10, boundary=0, position="identity", alpha=0.6, color="black") +
            scale_fill_manual(
                values=color_map,
                breaks=ordered_sources   # <- controls legend order
            ) +            
            scale_x_continuous(limits=(0,100)) +
            scale_y_continuous(breaks=range(0, 10+1), limits=(0, 10)) +
            labs(
                title="Agreement by Question Source",
                x="Agreement Percentage",
                y="Number of Reports"
            ) +
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
            #scale_x_continuous(limits=(0,100)) +
            scale_y_continuous(limits=(0,100)) +
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
        # Create a custom sort key for Question_Source: 0 for groundtruth, 1 for others
        disagg_df['Source_Order'] = disagg_df['Question_Source'].apply(
            lambda x: 0 if 'groundtruth' in x.lower() else 1
        )

        # Sort by Task_ID first, then by source_order
        disagg_df_sorted = disagg_df.sort_values(by=['Task_ID', 'Source_Order'])

        # Drop the temporary column before saving
        disagg_df_sorted = disagg_df_sorted.drop(columns=['Source_Order'])
        disagg_df_sorted.to_csv(os.path.join(output_dir, "agreement_report_level_by_source.csv"), index=False)

        aggregated_stats = pd.DataFrame({
            "Mean_Agreement": [mean_overall],
            "Std_Deviation": [std_overall]
        })
        aggregated_stats.to_csv(os.path.join(output_dir, "agreement_report_level_aggregated.csv"), index=False)

        print(f"Mean agreement: {mean_overall:.1f}%, Std Dev: {std_overall:.1f}%")

    except Exception as e:
        print(f"Error generating ggplot-style agreement plots: {e}")


def traditional_similarity_metrics(
    input_path="../data/benchmark/benchmark.csv",
    output_dir="../../output/",
    sts_model="dmlls/all-mpnet-base-v2-negation",
    biobert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    ground_truth_col="Ground Truth",
    compare_cols=None
):
    """
    Compute semantic textual similarity (STS), BERTScore, ROUGE and BLEU between
    generated reports and a ground truth report.

    Args:
        input_path: str
            Path to CSV or Excel file containing model outputs.
        output_dir: str
            Directory where results will be saved.
        model_name: str
            SentenceTransformer model name for STS.
        ground_truth_col: str
            Name of the ground truth column.
        compare_cols: list of str
            List of columns to compare against the ground truth.

    Returns:
        (DataFrame with scores, Summary DataFrame)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Loading data from {input_path}...")

        # --- Load data ---
        if input_path.endswith(".xlsx"):
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path)

        # --- Validate columns ---
        if ground_truth_col not in df.columns:
            raise ValueError(f"Ground truth column '{ground_truth_col}' not found in data.")

        if compare_cols is None:
            compare_cols = [c for c in df.columns if c != ground_truth_col]

        missing = [c for c in compare_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing comparison columns: {missing}")

        # Fill missing text with empty strings
        for c in [ground_truth_col] + compare_cols:
            df[c] = df[c].fillna("")

        print(f"Comparing reports: {compare_cols} against '{ground_truth_col}'")

        # --- Semantic Textual Similarity (STS) ---
        print("Computing Sentence Similarity (STS)...")
        sts_model = SentenceTransformer(sts_model)

        gt_emb = sts_model.encode(df[ground_truth_col], convert_to_tensor=True)
        for col in compare_cols:
            emb = sts_model.encode(df[col], convert_to_tensor=True)
            df[f"STS_{col}_vs_{ground_truth_col}"] = util.cos_sim(emb, gt_emb).diagonal().cpu().numpy()


        # --- 2. BioBERT (PubMedBERT) Semantic Similarity
        print("Computing BioBERT-based Semantic Similarity...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(biobert_model)
        model = AutoModel.from_pretrained(biobert_model).to(device)
        model.eval()

        def embed_long_texts(texts, tokenizer, model, device, max_len=512, stride=50, batch_size=4):
            """Embed long texts using sliding windows and mean pooling."""
            all_embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = []
                for text in batch_texts:
                    tokens = tokenizer.encode(text, add_special_tokens=True)
                    embeddings = []
                    start = 0
                    while start < len(tokens):
                        end = min(start + max_len, len(tokens))
                        input_ids = torch.tensor([tokens[start:end]]).to(device)
                        attention_mask = torch.ones_like(input_ids).to(device)
                        with torch.no_grad():
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            chunk_emb = outputs.last_hidden_state.mean(dim=1)
                            embeddings.append(chunk_emb.cpu())
                        if end == len(tokens):
                            break
                        start += max_len - stride
                    text_emb = torch.mean(torch.stack(embeddings), dim=0)
                    batch_embeddings.append(text_emb)
                all_embeddings.append(torch.cat(batch_embeddings, dim=0))
            return torch.cat(all_embeddings, dim=0).numpy()

        print("Embedding Ground Truth...")
        gt_embeddings = embed_long_texts(df[ground_truth_col].tolist(), tokenizer, model, device)

        for col in compare_cols:
            print(f"Embedding {col}...")
            model_embeddings = embed_long_texts(df[col].tolist(), tokenizer, model, device)
            sims = [
                cosine_similarity(gt_embeddings[i].reshape(1, -1), model_embeddings[i].reshape(1, -1))[0][0]
                for i in range(len(df))
            ]
            df[f"BioBERTScore_{col}_vs_{ground_truth_col}"] = sims

        # --- ROUGE-L (textual overlap) ---
        print("Computing ROUGE-L scores...")
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        def rougeL(hyp, ref):
            return rouge.score(ref, hyp)["rougeL"].fmeasure

        for col in compare_cols:
            df[f"ROUGE_{col}_vs_{ground_truth_col}"] = [
                rougeL(h, r) for h, r in zip(df[col], df[ground_truth_col])
            ]

        # --- BLEU Score (syntactic similarity) ---

        print("Computing BLEU scores...")
        smoothie = SmoothingFunction().method4

        def compute_bleu(candidate, reference):
            candidate_tokens = candidate.split()
            reference_tokens = reference.split()
            try:
                return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
            except ZeroDivisionError:
                return 0.0

        for col in compare_cols:
            df[f"BLEU_{col}_vs_{ground_truth_col}"] = [
                compute_bleu(h, r) for h, r in zip(df[col], df[ground_truth_col])
            ]
        # "BERTScore_"
        # --- Step 4: Summary Statistics ---
        metric_cols = [c for c in df.columns if any(k in c for k in ["STS_", "BioBERTScore_", "ROUGE_", "BLEU_"])]
        summary = df[metric_cols].describe().T.round(3)

        print("\n===Summary of Semantic & Biomedical Similarity ===")
        print(summary)

        # --- Save outputs ---
        df.to_csv(os.path.join(output_dir, "semantic_similarity_pertask.csv"), index=False)
        summary.to_csv(os.path.join(output_dir, "semantic_similarity_overall.csv"))

        print(f"\n Results saved to {output_dir}")

        return df, summary

    except Exception as e:
        print(f"Error in semantic similarity pipeline: {e}")
        return None, None
    

def compute_agreement_summary(
    benchmark_path: str,
    experiment_id: str,
    base_dir: str,
    report_type: str = "llm(gpt-4o)",
) -> pd.DataFrame:

    # -------- Load and preprocess benchmark --------
    bench = pd.read_csv(benchmark_path)

    # Clean Task_ID
    bench["Task_ID"] = (
        bench["ID"]
        .astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(r"0+$", "", regex=True)
        .str.rstrip(".")
        .astype(int)
    )

    # Extract year
    if "Date Published" not in bench.columns:
        raise KeyError("'Date Published' column not found in benchmark file.")
    bench["Year_Published"] = pd.to_datetime(
        bench["Date Published"], errors="coerce"
    ).dt.year

    # -------- Variables to merge + aggregate --------
    benchmark_vars = {
        "Number of Steps": "Number of Steps",
        "Difficulty: 1 (Easy) - 2 (Med) - 3 (Hard)": "Difficulty (Steps)",
        "Difficulty-Text: 1 (Easy) - 2 (Med) - 3 (Hard)": "Difficulty (Text)",
        "Year_Published": "Year_Published"

    }
    # Keep only those columns that actually exist in the benchmark
    existing_vars = {k: v for k, v in benchmark_vars.items() if k in bench.columns}

    # -------- Load agreement data --------
    agreement_path = (
        Path(base_dir)
        / "experiments"
        / experiment_id
        / "analysis"
        / f"groundtruth_vs_{report_type}"
        / "agreement_report_level_by_source.csv"
    )
    agreement = pd.read_csv(agreement_path)

    # Merge benchmark fields
    agreement = agreement.merge(
        bench[["Task_ID"] + list(existing_vars.keys())],
        how="left",
        on="Task_ID"
    )

    # -------- Helper function for aggregation --------
    def aggregate(var):
        """
        Aggregates by variable and question source.
        Returns: DataFrame with columns:
        Question_Source | Variable | Value | Mean_Agreement_Percentage
        """
        var_label = existing_vars[var]

        # By-source aggregation
        by_source = (
            agreement.groupby(["Question_Source", var], as_index=False)["Agreement_Percentage"]
            .mean()
            .rename(columns={
                "Agreement_Percentage": "Mean_Agreement_Percentage",
                var: "Value"
            })
        )
        by_source["Variable"] = var_label

        # Overall (across all sources)
        overall = (
            agreement.groupby([var], as_index=False)["Agreement_Percentage"]
            .mean()
            .rename(columns={
                "Agreement_Percentage": "Mean_Agreement_Percentage",
                var: "Value"
            })
        )
        overall["Variable"] = var_label
        overall["Question_Source"] = "Overall"

        return pd.concat([by_source, overall], ignore_index=True)

    # -------- Aggregate all variables dynamically --------
    summaries = [aggregate(var) for var in existing_vars]

    result = pd.concat(summaries, ignore_index=True)

    # -------- Final formatting --------
    result["Report_Type"] = report_type

    result = result[
        ["Report_Type", "Question_Source", "Variable", "Value", "Mean_Agreement_Percentage"]
    ].sort_values(
        ["Report_Type", "Variable", "Question_Source", "Value"]
    ).reset_index(drop=True)

    return result


def compute_all_reports(
    benchmark_path: str,
    experiment_id: str,
    report_types: list[str],
    base_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = []
    overall_stats = []

    for rtype in report_types:
        print(f"Processing report_type: {rtype} ...")
        df = compute_agreement_summary(
            benchmark_path=benchmark_path,
            experiment_id=experiment_id,
            report_type=rtype,
            base_dir=base_dir
        )
        combined.append(df)

        # --- Load raw agreement ---
        agreement_path = (
            Path(base_dir)
            / "experiments"
            / experiment_id
            / "analysis"
            / f"groundtruth_vs_{rtype}"
            / "agreement_report_level_by_source.csv"
        )

        if not agreement_path.exists():
            print(f" Missing file for {rtype}, skipping overall stats.")
            continue

        agreement = pd.read_csv(agreement_path)

        # --- Overall mean agreement per Question_Source ---
        overall = (
            agreement.groupby("Question_Source", as_index=False)["Agreement_Percentage"]
            .mean()
            .rename(columns={"Agreement_Percentage": "Overall_Mean_Agreement"})
        )
        overall["Report_Type"] = rtype

        # --- Compute % I don't know from JSONs ---
        idk_df = compute_percent_idontknow(experiment_id, rtype, base_dir)
        overall = overall.merge(idk_df, on=["Report_Type", "Question_Source"], how="left")
        overall_stats.append(overall)

    combined_df = pd.concat(combined, ignore_index=True)
    overall_df = pd.concat(overall_stats, ignore_index=True)

    # --- Melt overall to long format ---
    overall_df = overall_df.melt(
        id_vars=["Question_Source", "Report_Type"],
        value_vars=["Overall_Mean_Agreement", "Percent_IDontKnow"],
        var_name="Variable",
        value_name="Value"
    ).sort_values(
        by=["Report_Type", "Variable", "Question_Source"]
    ).reset_index(drop=True)

    # --- Save both dataframes ---
    analysis_path = Path(base_dir) / "experiments" / experiment_id / "analysis"
    analysis_path.mkdir(parents=True, exist_ok=True)

    combined_file = analysis_path / "summary_by_variable.csv"
    overall_file = analysis_path / "overall_summary.csv"

    combined_df.to_csv(combined_file, index=False)
    overall_df.to_csv(overall_file, index=False)

    print(f"Saved variable-level summary to {combined_file}")
    print(f"Saved overall summary to {overall_file}")


def compute_percent_idontknow(experiment_id: str, report_type: str, base_dir) -> pd.DataFrame:
    """
    Compute % of "I don't know" predictions (prediction=='E') for a given report type.
    Returns a DataFrame with columns: Report_Type, Question_Source, Percent_IDontKnow
    """
    folder = Path(base_dir) / "experiments" / experiment_id / "answers" / f"groundtruth_vs_{report_type}"

    records = []
    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            filepath = folder / filename
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data['filename'] = filename
                        records.append(data)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                item['filename'] = filename
                                records.append(item)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Could not read {filename}: {e}")

    df = pd.DataFrame(records)

    if df.empty or 'prediction' not in df.columns or 'question_source' not in df.columns:
        return pd.DataFrame(columns=["Report_Type", "Question_Source", "Percent_IDontKnow"])

    results = []
    for source in df['question_source'].unique():
        subset = df[df['question_source'] == source]
        percent = (subset['prediction'] == 'E').mean() * 100
        results.append({"Report_Type": report_type, "Question_Source": source, "Percent_IDontKnow": percent})

    return pd.DataFrame(results)
