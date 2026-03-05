import numpy as np
from collections import defaultdict
from rf100vl.util import get_basename, get_category

import os
import json
import glob
import pandas as pd
import argparse
from pathlib import Path

from textwrap import dedent

    
# # Dictionary of mAP values for different datasets
# res = {
#     'mAP_actions-zzid2-zb1hq-fsod-amih': 24.064278521404297,
#     'mAP_trail-camera-fsod-egos': 56.7836977264109,
#     'mAP_paper-parts-fsod-rmrg': 37.17713035390103,
#     'mAP_lacrosse-object-detection-fsod-uxkt': 18.044207431906667,
#     'mAP_flir-camera-objects-fsod-tdqp': 17.605628602665618,
#     'mAP_the-dreidel-project-anzyr-fsod-zejm': 37.85522417418126,
#     'mAP_water-meter-jbktv-7vz5k-fsod-ftoz': 26.56982401633573,
#     'mAP_orionproducts-vtl2z-fsod-puhv': 25.455224670274234,
#     'mAP_aerial-airport-7ap9o-fsod-ddgc': 31.34755646407234,
#     'mAP_wildfire-smoke-fsod-myxt': 34.408241015044446,
#     'mAP_soda-bottles-fsod-haga': 15.953955247726043,
#     'mAP_all-elements-fsod-mebv': 25.076353709933763,
#     'mAP_dentalai-i4clz-fsod-fsuo': 9.094968428106615,
#     'mAP_wb-prova-stqnm-fsod-rbvg': 51.5895982439329,
#     'mAP_aquarium-combined-fsod-gjvb': 32.592390961689794,
#     'mAP_x-ray-id-zfisb-fsod-dyjv': 38.44694385039646,
#     'mAP_defect-detection-yjplx-fxobh-fsod-amdi': 46.201750742340195,
#     'mAP_new-defects-in-wood-uewd1-fsod-tffp': 25.96925844322867,
#     'mAP_gwhd2021-fsod-atsv': 16.79637417893668,
#     'mAP_recode-waste-czvmg-fsod-yxsw': 32.2488722804605,
#     'mAP': 30.164073953147415
# }

def groupResults(res, eval_type: str, results_root_dir: str, consolidated_results_path: str, caption_suffix: str = ""):
    # Remove the "mAP_" prefix from all keys
    # res = {k.replace("mAP_", ""): v for k, v in res.items()}

    # Extract the overall mAP value
    # mAP = res.pop("mAP")
    mAP = res.pop("AVERAGE")

    ignore_flag = False
    #Check 20 datasets
    # if len(res) != 20:
    #     if "paper-parts" not in res:
    #         print("Warning: 'paper-parts' dataset missing from results. Adding with mAP=0.0")
    #         # res["paper-parts"] = 0.0
    #         res["paper-parts"] = 0.03757084180854858 #Qwen2.5-7B
    #         # res["paper-parts"] = 0.03721320533192437 #Qwen2.5-72B
    #         # res["paper-parts"] = 0.033762771089204975 #Qwen3-8B
    #         # res["paper-parts"] = 0.0634269094408435 #Qwen3-235B
    #         ignore_flag = True

    assert len(res) == 20, f"Expected 20 datasets, but got {len(res)}"

    #Calculate mean
    mean_map = np.mean(list(res.values()))
    print(f"Mean mAP across all datasets: {mean_map*100:.4f}")
    assert mAP == mean_map or ignore_flag, f"mAP value {mAP} does not match calculated mean {mean_map}"

    sum_map = np.sum(list(res.values()))
    print(f"Sum mAP across all datasets: {sum_map*100:.4f}")
    avgOn20 = sum_map / 20.0
    print(f"Average mAP across 20 datasets: {avgOn20*100:.1f}")

    # Group datasets by category and collect their mAP values
    finals = defaultdict(list)
    for key, value in res.items():
        # cat = get_category(get_basename(key))
        cat = get_category(key)
        finals[cat].append(value)

    # # Print the mean mAP for each category
    # print(f"Total mAP: {mAP*100:.4f}")
    # for k, v in finals.items():
    #     print(f"{k}: {np.mean(v)*100:.4f}")
    print(f"Total mAP: {mAP*100:.1f}")
    for k, v in finals.items():
        print(f"{k}: {np.mean(v)*100:.1f}")

    # Save the consolidated results to a CSV file
    output_records = []
    for category, values in finals.items():
        record = {
            'category': category,
            'mean_mAP': np.mean(values)*100,
            'num_datasets': len(values)
        }
        output_records.append(record)

    df = pd.DataFrame(output_records)
    df.to_csv(os.path.join(consolidated_results_path, f'consolidated_results_{eval_type}.csv'), index=False)

    #Save as text file as well
    with open(os.path.join(consolidated_results_path, f'consolidated_results_{eval_type}.txt'), 'w') as f:
        f.write(f"Total mAP: {mAP*100:.1f}\n")
        for k, v in finals.items():
            f.write(f"{k}: {np.mean(v)*100:.1f}\n")
    
    #Save all individual dataset results as well
    with open(os.path.join(consolidated_results_path, f'consolidated_results_{eval_type}_individual.txt'), 'w') as f:
        for key, value in res.items():
            f.write(f"{key}: {value*100:.4f}\n")

    print("Saving consolidated results to CSV:", os.path.join(consolidated_results_path, f'consolidated_results_{eval_type}.csv'))




def consolidate_evaluation_results(results_root_dir: str, output_csv_path: str):
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
    # exp_group_pattern = os.path.join(results_root_dir, 'rf100vl_zeroshot_onIPT')  # Specify the exact group if needed
    exp_group_pattern = results_root_dir
    exp_group_dirs = glob.glob(exp_group_pattern)

    print(f"Found {len(exp_group_dirs)} experiment groups matching: {exp_group_pattern}")

    metric_keys = ['AP_50_95', 'AP_50', 'AP_75', 'AP_S', 'AP_M', 'AP_L',
                   'AR_1', 'AR_10', 'AR_100', 'AR_S', 'AR_M', 'AR_L']
    

    metric_keys2 = ['AP_50_95', 'AP_50', 'AP_75', 'AP_small', 'AP_medium', 'AP_large',
                   'AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large']

    for exp_group_dir in exp_group_dirs:
        # Within each group, find all specific run directories (e.g., rf20_iterCat_v1a)
        # run_dirs = [d.path for d in os.scandir(exp_group_dir) if d.is_dir()]
        # print(f"Processing experiment group: {Path(exp_group_dir).name} with {len(run_dirs)} runs.")

        run_dir = exp_group_dir
        
        eval_type = 'model'

        # for run_dir in run_dirs:


        # conf_type = ["model", "rank", "vqa", "siglip", "vqa_nocontext", "vqa_defaultInstr",]

        conf_type = ["model", "rank", "vqa", "siglip", "vqa_nocontext", "vqa_defaultInstr",]
        # conf_type = ["vqa_nocontext", "vqa_defaultInstr",]

        # for eval_type in conf_type:
            # Look for the evaluation JSON files
            # eval_json_pattern = os.path.join(run_dir, 'final_instruction_eval', 'evaluations', '*', 'evaluation_*.json') #default - implies with no_instructions
            # eval_json_pattern = os.path.join(run_dir, 'final_instruction_eval', 'evaluations', 'vqa_nms0.5', 'evaluation_*.json') #default - implies with no_instructions
        
        # if eval_type == "model":
        #     eval_json_pattern = os.path.join(run_dir, 'final_instruction_eval', 'evaluations', 'rank', 'evaluation_*.json') #default - implies with no_instructions
        # else:
        #     eval_json_pattern = os.path.join(run_dir, 'final_instruction_eval', 'evaluations', eval_type, 'evaluation_*.json') #default - implies with no_instructions
        eval_json_pattern = os.path.join(run_dir, 'evaluation_*.json') #default - implies with no_instructions
        eval_files = glob.glob(eval_json_pattern)

        if not eval_files:
            continue

        run_all_results = []
        run_metrics_dict = {}
        run_name = f"{Path(exp_group_dir).name}/{Path(run_dir).name}"
        print(f"  Processing run: {run_name} ({len(eval_files)} files)")

        for eval_file in eval_files:
            try:
                with open(eval_file, 'r') as f:
                    data = json.load(f)
                
                dataset_name = Path(eval_file).stem.replace('evaluation_', '')
                
                # # Store individual dataset result   
                # for k,v in data.items():
                k = 'model'
                v = data
            
                # if k != eval_type and (eval_type == "rank" and k != "ranking") and (k == "vqa" and "vqa" not in eval_type):
                #     continue

                v_dict = {}
                for i in range(len(v)):
                    # v_dict[metric_keys[i]] = v[i]
                    v_dict[metric_keys[i]] = v[metric_keys2[i]]
                
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


            # for k,v in data.items():

            k = 'model'
            v = data

            #     if k != eval_type and (eval_type == "rank" and k != "ranking") and (k == "vqa" and "vqa" not in eval_type):
            #         continue


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

            # save_evaluation_tables(run_all_results, eval_type, run_dir, output_csv_path, caption_suffix = f"\\\\Experiment Run: {run_name.replace("_", "\\_")}")
            df = pd.DataFrame(run_all_results)
            
            #Filter expermiment run == eval_type
            # if k == "vqa" and "vqa" in eval_type:
            #     df = df[df['experiment_run'].str.contains(k)]
            # else:
            #     df = df[df['experiment_run'].str.contains(eval_type)]

            df = df[df['experiment_run'].str.contains(eval_type)]

            #Drop all columns except dataset and AP_50_95
            df = df[['dataset', 'AP_50_95']]

            #Convert to dict with key as dataset and value as AP_50_95
            df = df.set_index('dataset')['AP_50_95']
            df_dict = df.to_dict()

            
            # groupResults(df_dict, eval_type, results_root_dir, output_csv_path, caption_suffix = f"\\\\Experiment Run: {run_name.replace("_", "\\_")}")
            consol_results_dir = os.path.join(run_dir, "consolidated_results")
            os.makedirs(consol_results_dir, exist_ok=True)
            groupResults(df_dict, eval_type, results_root_dir, consol_results_dir, caption_suffix = f"Experiment Run: {run_name}")

    # if not all_results:
    #     print("No evaluation files found. The output CSV will be empty.")
    #     return


    # save_evaluation_tables(all_results, eval_type, results_root_dir, output_csv_path)



# cmd:
# // "--results_dir", "results/final_consolidated_results/rf-20-vl-benchmark/results/eccv26/rf100vl_default/Qwen3-VL-30B-A3B-Instruct/",
# // "--results_dir", "results/final_consolidated_results/rf-20-vl-benchmark/results/eccv26/rf100vl_IPT/Qwen2.5-VL-7B-Instruct/",
# // "--results_dir", "results/final_consolidated_results/rf-20-vl-benchmark/results/eccv26/rf100vl_IPT/Qwen2.5-VL-72B-Instruct/",
# // "--results_dir", "results/final_consolidated_results/rf-20-vl-benchmark/results/eccv26/rf100vl_IPT/Qwen3-VL-8B-Instruct/",
# // "--results_dir", "results/final_consolidated_results/rf-20-vl-benchmark/results/eccv26/rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/",
# // "--results_dir", "results/final_consolidated_results/dspy-baselines/results/eccv26/gepa/Qwen3-VL-30B-A3B-Instruct/rf20_gepa_REF_LM_qwen_singleclass_rankScore/imputed_results/",
# // "--results_dir", "results/final_consolidated_results/dspy-baselines/results/eccv26/mipro/Qwen3-VL-30B-A3B-Instruct/rf20_mipro_REF_LM_qwen_singleclass_rankScore/imputed_results/",
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate evaluation results from multiple experiment runs.")
    # parser.add_argument(
    #     '--eval_type',
    #     type=str,
    #     # default='noinstr',
    #     # default='vqa_no_nms',
    #     default='model',
    #     # default='rank',
    #     # default='vqa',
    #     # default='siglip',
    #     # default='vqa_defaultInstr',
    #     # default='vqa_nocontext',
    #     # default='default',
    #     help='The evaluation type: noinstr, default, or few_shot.'
    # )
    parser.add_argument(
        '--results_dir',
        type=str,
        # default='/home/grg/Research/VLMattributeClassifier/results',
        default='./results/final_results/rf100vl_IPT',
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

    consolidate_evaluation_results(args.results_dir, args.output_csv)
