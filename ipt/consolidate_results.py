import os
import json
import glob
import pandas as pd
import argparse
from pathlib import Path

from textwrap import dedent
    
def save_evaluation_tables(all_results, eval_type: str, results_root_dir: str, output_csv_path: str, caption_suffix: str = ""):
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns for clarity
    cols = ['experiment_run', 'dataset'] + [c for c in results_df.columns if c not in ['experiment_run', 'dataset']]
    results_df = results_df[cols]

    # Ensure output directory exists
    output_csv_path = os.path.join(results_root_dir, output_csv_path.split('.')[0] + f"_{eval_type}.csv")
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv_path, index=False, float_format='%.6f')
    print(f"\nSuccessfully consolidated results to: {output_csv_path}")

    # --- LaTeX Table Generation and Saving ---
    # Create a copy for LaTeX formatting to avoid changing the original DataFrame
    latex_df = results_df.copy()
    
    # Multiply metric columns by 100 for percentage representation
    metric_cols = [col for col in latex_df.columns if col.startswith('AP') or col.startswith('AR')]
    latex_df[metric_cols] = latex_df[metric_cols] * 100
    
    latex_df['dataset'] = latex_df['dataset'].str.replace('_', r'\_', regex=False)

    latex_output_path = Path(output_csv_path).with_suffix('.tex')
    all_latex_content = []

    # Generate Combined Table
    try:
        metrics_to_display = ['AP_50_95', 'AR_1']
        # Create a pivot table with a multi-level column index
        pivot_df = latex_df.pivot(index='dataset', columns='experiment_run', values=metrics_to_display)

        # Sort the columns to group experiments under each metric
        pivot_df.sort_index(axis=1, level=0, inplace=True)

        # Escape underscores in the column names (level 0: metrics, level 1: experiments)
        new_columns = []
        for col in pivot_df.columns:
            metric_name, experiment_name = col
            escaped_metric = metric_name.replace('_', r'\_')
            escaped_experiment = experiment_name.replace('_', r'\_')
            new_columns.append((escaped_metric, escaped_experiment))
        pivot_df.columns = pd.MultiIndex.from_tuples(new_columns, names=pivot_df.columns.names)

        # Escape underscores in the index (dataset)
        # The values are already escaped from latex_df, but the index name is not.
        # Note: The index values are already escaped from latex_df['dataset']
        pivot_df.index.name = pivot_df.index.name.replace('_', r'\_')

        latex_table = pivot_df.to_latex(float_format="%.2f", caption=f"AP@50-95 and AR@1 scores across datasets and experiments.{caption_suffix}", label="tab:combined_results", multicolumn_format='c', escape=False)
        
        def formatTable(latex_table):
            # # Remove \begin{table}...\end{table} from pandas output
            # latex_str = latex_str.replace("\\begin{table}", "").replace("\\end{table}", "").strip()
            # latex_str = latex_str.replace("\n", "\n\t\t\t\t\t").strip()
            # # Wrap with \resizebox for fitting to text width
            # latex_str = dedent(f"""
            #     \\begin{{table*}}[htbp]
            #     \\centering
            #     \\resizebox{{\\textwidth}}{{!}}{{%
            #             {latex_str}
            #     }}
            #     \\end{{table*}}
            # """)


            # Remove outer table environment
            latex_table = latex_table.replace("\\begin{table}", "").replace("\\end{table}", "").strip()

            # Insert \resizebox after caption and label
            lines = latex_table.splitlines()
            for i, line in enumerate(lines):
                if "\\label" in line:
                    insert_idx = i + 1
                    break
            else:
                insert_idx = len(lines)

            # lines.insert(insert_idx, "    \\resizebox{\\textwidth}{!}{%")
            # lines.append("    }")

            # # Wrap in a single table environment
            # latex_table = dedent(f"""
            # \\begin{{table}}[htbp]
            #     \\centering
            # {chr(10).join(lines)}
            # \\end{{table}}
            # """)


            lines.insert(insert_idx, "\\resizebox{\\textwidth}{!}{%")
            lines.append("}")

            # Wrap in a single table environment
            latex_table = dedent(f"\
            \\begin{{table*}}[htbp]\n\
            \\centering\n\
            {chr(10).join(lines)}\n\
            \\end{{table*}}\
            ")
            
            latex_table = latex_table.replace("\n          ", "\n").replace("\n  ", "\n").strip()
            
            return latex_table

        latex_table = formatTable(latex_table)

        all_latex_content.append("="*20 + " Combined LaTeX Table " + "="*20)
        all_latex_content.append(latex_table)
        all_latex_content.append("="*64 + "\n")

        print("\n" + "="*20 + " Combined LaTeX Table " + "="*20)
        print(latex_table)
        print("="*64)

    except Exception as e:
        error_msg = f"Could not generate combined LaTeX table. Error: {e}"
        print(f"\n{error_msg}")
        all_latex_content.append(error_msg)

    # Generate Separate Tables
    try:
        # Pivot for AP_50_95
        # Use the modified latex_df which has values multiplied by 100
        pivot_ap_50_95 = latex_df.pivot(index='experiment_run', columns='dataset', values='AP_50_95')
        # Escape underscores in the index (experiment_run)
        pivot_ap_50_95.index = pivot_ap_50_95.index.str.replace('_', r'\_', regex=False)
        pivot_ap_50_95.index.name = pivot_ap_50_95.index.name.replace('_', r'\_')
        # Column names (datasets) are already escaped in latex_df

        latex_ap_50_95 = pivot_ap_50_95.to_latex(float_format="%.2f", caption=f"AP@50-95 scores for different experiment runs.{caption_suffix}", label="tab:ap5095", escape=False)
                
        latex_ap_50_95 = formatTable(latex_ap_50_95)

        print("\n" + "="*20 + " LaTeX Table for AP@50-95 " + "="*20)
        print(latex_ap_50_95)
        all_latex_content.append("="*20 + " LaTeX Table for AP@50-95 " + "="*20)
        all_latex_content.append(latex_ap_50_95)
        all_latex_content.append("="*64 + "\n")

        # Pivot for AR_1
        pivot_ar_1 = latex_df.pivot(index='experiment_run', columns='dataset', values='AR_1')
        # Escape underscores in the index (experiment_run)
        pivot_ar_1.index = pivot_ar_1.index.str.replace('_', r'\_', regex=False)
        pivot_ar_1.index.name = pivot_ar_1.index.name.replace('_', r'\_')
        # Column names (datasets) are already escaped in latex_df

        latex_ar_1 = pivot_ar_1.to_latex(float_format="%.2f", caption=f"AR@1 scores for different experiment runs.{caption_suffix}", label="tab:ar1", escape=False)
        
        latex_ar_1 = formatTable(latex_ar_1)

        print("\n" + "="*20 + " LaTeX Table for AR@1 " + "="*20)
        print(latex_ar_1)
        print("="*62)
        all_latex_content.append("\n" + "="*20 + " LaTeX Table for AR@1 " + "="*20)
        all_latex_content.append(latex_ar_1)
        all_latex_content.append("="*62)
        
    except Exception as e:
        error_msg = f"Could not generate separate LaTeX tables. Error: {e}"
        print(f"\n{error_msg}")
        all_latex_content.append(error_msg)

    # with open(latex_output_path, 'w') as f:
    #     f.write('\n'.join(all_latex_content))
    # print(f"Successfully saved LaTeX tables to: {latex_output_path}")


    # Generate Combined Table
    try:
        metrics_to_display = ['AP_50_95', 'AR_1']
        # Use the modified latex_df which has values multiplied by 100
        pivot_df = latex_df.pivot(index='experiment_run', columns='dataset', values=metrics_to_display)

        # Swap the column levels to have dataset as the top level
        pivot_df = pivot_df.swaplevel(0, 1, axis=1)
        # Sort the columns to group metrics under each dataset
        pivot_df.sort_index(axis=1, level=0, inplace=True)

        # Escape underscores in the second level of the column names (the metrics)
        new_columns = []
        for col in pivot_df.columns:
            dataset_name, metric_name = col
            escaped_metric_name = metric_name.replace('_', r'\_')
            new_columns.append((dataset_name, escaped_metric_name))
        pivot_df.columns = pd.MultiIndex.from_tuples(new_columns, names=pivot_df.columns.names)

        # Escape underscores in the index (experiment_run)
        pivot_df.index = pivot_df.index.str.replace('_', r'\_', regex=False)
        pivot_df.index.name = pivot_df.index.name.replace('_', r'\_')

        latex_table = pivot_df.to_latex(float_format="%.2f", caption=f"AP@50-95 and AR@1 scores across datasets and experiments.{caption_suffix}", label="tab:combined_results", multicolumn_format='c', escape=False)
        
        latex_table = formatTable(latex_table)

        all_latex_content.append("="*20 + " Combined LaTeX Table " + "="*20)
        all_latex_content.append(latex_table)
        all_latex_content.append("="*64 + "\n")

        print("\n" + "="*20 + " Combined LaTeX Table " + "="*20)
        print(latex_table)
        print("="*64)

    except Exception as e:
        error_msg = f"Could not generate combined LaTeX table. Error: {e}"
        print(f"\n{error_msg}")
        all_latex_content.append(error_msg)

    # Generate another Combined Table with experiments as rows
    try:
        metrics_to_display = ['AP_50_95', 'AR_1']
        # Use the modified latex_df which has values multiplied by 100
        pivot_df_exp_rows = latex_df.pivot(index='experiment_run', columns='dataset', values=metrics_to_display)

        # Swap the column levels to have dataset as the top level
        pivot_df_exp_rows = pivot_df_exp_rows.swaplevel(0, 1, axis=1)
        # Sort the columns to group metrics under each dataset
        pivot_df_exp_rows.sort_index(axis=1, level=0, inplace=True)

        # Escape underscores in the second level of the column names (the metrics)
        new_columns_exp = []
        for col in pivot_df_exp_rows.columns:
            dataset_name, metric_name = col
            escaped_metric_name = metric_name.replace('_', r'\_')
            new_columns_exp.append((dataset_name, escaped_metric_name))
        pivot_df_exp_rows.columns = pd.MultiIndex.from_tuples(new_columns_exp, names=pivot_df_exp_rows.columns.names)

        # Escape underscores in the index (experiment_run)
        pivot_df_exp_rows.index = pivot_df_exp_rows.index.str.replace('_', r'\_', regex=False)
        pivot_df_exp_rows.index.name = pivot_df_exp_rows.index.name.replace('_', r'\_')

        latex_table_exp_rows = pivot_df_exp_rows.to_latex(float_format="%.2f", caption=f"AP@50-95 and AR@1 scores with experiments as rows.{caption_suffix}", label="tab:combined_results_exp_rows", multicolumn_format='c', escape=False)
                        
        latex_table_exp_rows = formatTable(latex_table_exp_rows)

        all_latex_content.append("="*20 + " Combined LaTeX Table (Experiments as Rows) " + "="*20)
        all_latex_content.append(latex_table_exp_rows)
        all_latex_content.append("="*84 + "\n")

        print("\n" + "="*20 + " Combined LaTeX Table (Experiments as Rows) " + "="*20)
        print(latex_table_exp_rows)
        print("="*84)

    except Exception as e:
        error_msg = f"Could not generate combined LaTeX table with experiments as rows. Error: {e}"
        print(f"\n{error_msg}")
        all_latex_content.append(error_msg)


    with open(latex_output_path, 'w') as f:
        f.write('\n'.join(all_latex_content))
    print(f"Successfully saved LaTeX tables to: {latex_output_path}")


def consolidate_evaluation_results(eval_type: str, results_root_dir: str, output_csv_path: str):
    """
    Traverses experiment directories, reads evaluation JSON files, calculates
    average metrics for each experiment, and saves a consolidated CSV.

    Args:
        results_root_dir (str): The root directory containing all experiment runs
                                (e.g., '/home/grg/Research/VLMattributeClassifier/results').
        output_csv_path (str): The path to save the final consolidated CSV file.
    """
    all_results = []
    
    # Find all experiment group directories (e.g., rf100vl_tri0-11)
    # exp_group_pattern = os.path.join(results_root_dir, 'rf100vl_tri*')
    # exp_group_pattern = os.path.join(results_root_dir, 'rf100vl_tri0-11_v1')  # Specify the exact group if needed
    # exp_group_pattern = os.path.join(results_root_dir, 'rf100vl_new_org')  # Specify the exact group if needed
    # exp_group_pattern = os.path.join(results_root_dir, 'rf100vl_new_v1')  # Specify the exact group if needed
    # exp_group_pattern = os.path.join(results_root_dir, 'rf100vl_fixedPadBug')  # Specify the exact group if needed
    # exp_group_pattern = os.path.join(results_root_dir, 'rf100vl_zeroshot')  # Specify the exact group if needed
    exp_group_pattern = os.path.join(results_root_dir, 'rf100vl_zeroshot_onIPT')  # Specify the exact group if needed
    exp_group_dirs = glob.glob(exp_group_pattern)

    print(f"Found {len(exp_group_dirs)} experiment groups matching: {exp_group_pattern}")

    metric_keys = ['AP_50_95', 'AP_50', 'AP_75', 'AP_S', 'AP_M', 'AP_L',
                   'AR_1', 'AR_10', 'AR_100', 'AR_S', 'AR_M', 'AR_L']

    for exp_group_dir in exp_group_dirs:
        # Within each group, find all specific run directories (e.g., rf20_iterCat_v1a)
        run_dirs = [d.path for d in os.scandir(exp_group_dir) if d.is_dir()]
        
        for run_dir in run_dirs:
            # Look for the evaluation JSON files
            # eval_json_pattern = os.path.join(run_dir, 'evaluations', 'default', 'evaluation_*.json') #default - implies with instructions
            # eval_json_pattern = os.path.join(run_dir, 'evaluations', eval_type, 'evaluation_*.json') #default - implies with no_instructions
            eval_json_pattern = os.path.join(run_dir, 'evaluations', '*', 'evaluation_*.json') #default - implies with no_instructions
            eval_files = glob.glob(eval_json_pattern)

            if not eval_files:
                continue

            run_all_results = []
            # run_metrics = []
            run_metrics_dict = {}
            run_name = f"{Path(exp_group_dir).name}/{Path(run_dir).name}"
            print(f"  Processing run: {run_name} ({len(eval_files)} files)")

            for eval_file in eval_files:
                try:
                    with open(eval_file, 'r') as f:
                        data = json.load(f)
                    
                    dataset_name = Path(eval_file).stem.replace('evaluation_', '')
                    
                    # Store individual dataset result
                    for k,v in data.items():
                        v_dict = {}
                        for i in range(len(v)):
                            v_dict[metric_keys[i]] = v[i]
                        
                        run_record = {
                            # 'experiment_run': f"{run_name}_{k}",
                            'experiment_run': f"{k}",
                            'dataset': dataset_name,
                            # v  # Unpack all metrics from the JSON
                            **v_dict

                        }

                        run_all_results.append(run_record)

                        record = {
                            'experiment_run': f"{run_name}_{k}",
                            # 'experiment_run': f"{k}",
                            'dataset': dataset_name,
                            # v  # Unpack all metrics from the JSON
                            **v_dict

                        }
                        all_results.append(record)
                        # run_metrics.append(v)

                        if k not in run_metrics_dict:
                            run_metrics_dict[k] = [v_dict]
                        else:
                            run_metrics_dict[k].append(v_dict)

                        # run_metrics.append(v_dict)

                except (json.JSONDecodeError, IOError) as e:
                    print(f"    - Warning: Could not read or parse {eval_file}. Error: {e}")

            # Calculate and add the average for the current run
            if run_metrics_dict:
                # avg_df = pd.DataFrame(run_metrics)
                # avg_metrics = avg_df.mean().to_dict()
                # avg_record = {
                #     'experiment_run': run_name,
                #     'dataset': 'AVERAGE',
                #     **avg_metrics
                # }
                # run_all_results.append(avg_record)
                # all_results.append(avg_record)

                for k,v in data.items():

                    avg_df = pd.DataFrame(run_metrics_dict[k])
                    avg_metrics = avg_df.mean().to_dict()

                    run_avg_record = {
                        # 'experiment_run': f"{run_name}_{k}",
                        'experiment_run': f"{k}",
                        'dataset': 'AVERAGE',
                        **avg_metrics
                    }
                    run_all_results.append(run_avg_record)
                    avg_record = {
                        'experiment_run': f"{run_name}_{k}",
                        # 'experiment_run': f"{k}",
                        'dataset': 'AVERAGE',
                        **avg_metrics
                    }
                    all_results.append(avg_record)

            save_evaluation_tables(run_all_results, eval_type, run_dir, output_csv_path, caption_suffix = f"\\\\Experiment Run: {run_name.replace("_", "\\_")}")

    if not all_results:
        print("No evaluation files found. The output CSV will be empty.")
        return


    save_evaluation_tables(all_results, eval_type, results_root_dir, output_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate evaluation results from multiple experiment runs.")
    parser.add_argument(
        '--eval_type',
        type=str,
        # default='noinstr',
        default='vqa',
        # default='default',
        help='The evaluation type: noinstr, default, or few_shot.'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        # default='/home/grg/Research/VLMattributeClassifier/results',
        default='./results',
        help='The root directory where all experiment results are stored.'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        # default='./consolidated_results.csv',
        default='consolidated_results.csv',
        help='Path to save the consolidated CSV file.'
    )
    args = parser.parse_args()

    consolidate_evaluation_results(args.eval_type, args.results_dir, args.output_csv)
