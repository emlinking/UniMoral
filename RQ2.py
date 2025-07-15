import os
os.environ['HF_HOME'] = "/shared/0/projects/code-switching/datasets"

from transformers import AutoTokenizer
from huggingface_hub import login
import argparse
import pandas as pd
import ast
from tqdm import tqdm
import json
import random
import numpy as np
import torch
from collections import defaultdict
torch.cuda.empty_cache()
import utils

with open("hf_token.txt", "r") as token_file:
    access_token = token_file.read().strip()
login(token = access_token)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

act2idx = {'a': 'a', 'b': 'b', '1': 'a', '2': 'b', 1: 'a', 2: 'b'}
idx2label = ['Deontological', 'Utilitarianism', 'Rights-based', 'Virtuous', 'Unsure']
label2idx = {lab:i for i,lab in enumerate(idx2label)}
idx2moral = ['Care', 'Equality', 'Proportionality', 'Loyalty', 'Authority', 'Purity']
idx2culture = ['Power Distance', 'Individualism', 'Motivation', 'Uncertainty Avoidance', 'Long Term Orientation', 'Indulgence']

def calculate_metrics(true_values, pred_values):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total_true = defaultdict(int)

    for ground_truth, prediction in zip(true_values, pred_values):
        found = False
        for gt in ground_truth:
            total_true[gt] += 1
            if prediction == gt:
                tp[gt] += 1
                found = True
        if not found:
            fp[prediction] += 1
            for gt in ground_truth:
                fn[gt] += 1

    precision = {}
    recall = {}
    f1_score = {}

    all_classes = set(i for sublist in true_values for i in sublist)
    all_classes.update(pred_values)

    for cls in all_classes:
        precision[cls] = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        recall[cls] = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        if precision[cls] + recall[cls] > 0:
            f1_score[cls] = 2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls])
        else:
            f1_score[cls] = 0

    total_instances = sum(total_true.values())
    weighted_f1_score = sum((f1_score[cls] * total_true[cls] for cls in all_classes)) / total_instances

    return precision, recall, f1_score, weighted_f1_score

def dict_to_list_of_vectors(data):
    data = ast.literal_eval(data)
    return list(data.values())

def read_data_RQ2(data_file_long):
    print("Reading data...")
    data = pd.read_csv(data_file_long)
    data = data[['Scenario_id', 'Annotator_id', 'Scenario', 'Possible_actions', 'Selected_action', 'Action_criteria', 'Moral_values', 'Cultural_values', 'Annotator_self_description']]
    print("Data read -- ", len(data), "\n", data.columns, "\n", data.head())

    main_data = []
    for i in tqdm(range(len(data))):
        moral_vector = dict_to_list_of_vectors(data['Moral_values'][i])
        culture_vector = dict_to_list_of_vectors(data['Cultural_values'][i])

        moral_idx = sorted(range(len(moral_vector)), key=lambda i: moral_vector[i], reverse=True)
        culture_idx = sorted(range(len(culture_vector)), key=lambda i: culture_vector[i], reverse=True)

        if type(ast.literal_eval(data['Possible_actions'][i])[0]) != str:
            sel_act = act2idx[data['Selected_action'][i]] + ": " + ast.literal_eval(data['Possible_actions'][i])[0][data['Selected_action'][i]-1]
        else:
            sel_act = act2idx[data['Selected_action'][i]] + ": " + ast.literal_eval(data['Possible_actions'][i])[data['Selected_action'][i]-1]

        action_criteria = data['Action_criteria'][i]
        action_criteria_list = [int(x) for x in ast.literal_eval(action_criteria)]
        action_criteria_list = np.array(action_criteria_list)
        action_criteria_list = np.where(action_criteria_list == action_criteria_list.max())[0]
        action_criteria_list = [idx2label[x] for x in action_criteria_list]
            
        this_inst = {
            'scenario_id': data['Scenario_id'][i],
            'annotator_id': data['Annotator_id'][i],
            'scenario': data['Scenario'][i],
            'actions': data['Possible_actions'][i],
            'action_selected': sel_act,
            'gt': action_criteria_list,
            'desc': data['Annotator_self_description'][i],
            'moral_vector': moral_vector,
            'culture_vector': culture_vector,
            'moral_idx': [idx2moral[x] for x in moral_idx],
            'culture_idx': [idx2culture[x] for x in culture_idx]
        }
        main_data.append(this_inst)

    for inst in main_data:
        annotator = inst['annotator_id']
        annotator_data = data[data['Annotator_id'] == annotator]
        annotator_data = annotator_data[annotator_data['Scenario_id'] != inst['scenario_id']]
        try:
            annotator_data = annotator_data.sample(n=1, replace=True, random_state=42)
        except:
            annotator_data = data[data['Annotator_id'] == annotator]
            annotator_data = annotator_data.sample(n=1, replace=True, random_state=42)
        annotator_data = annotator_data.reset_index(drop=True)
        
        if type(ast.literal_eval(annotator_data['Possible_actions'][0])[0]) != str:
            sel_act = act2idx[annotator_data['Selected_action'][0]] + ": " + ast.literal_eval(annotator_data['Possible_actions'][0])[0][annotator_data['Selected_action'][0]-1]
        else:
            sel_act = act2idx[annotator_data['Selected_action'][0]] + ": " + ast.literal_eval(annotator_data['Possible_actions'][0])[annotator_data['Selected_action'][0]-1]
        
        action_criteria = annotator_data['Action_criteria'][0]
        action_criteria_list = [int(x) for x in ast.literal_eval(action_criteria)]
        action_criteria_list = np.array(action_criteria_list)
        print("action_criteria_list: ", action_criteria_list)
        print("np.where(action_criteria_list == action_criteria_list.max()): ", np.where(action_criteria_list == action_criteria_list.max()))
        action_criteria_list = np.where(action_criteria_list == action_criteria_list.max())[0]
        print("action_criteria_list = np.where(action_criteria_list == action_criteria_list.max())[0]: ", np.where(action_criteria_list == action_criteria_list.max())[0])
        action_criteria_list = ", ".join([idx2label[x] for x in action_criteria_list])

        inst['fs_data'] = {
            'fs_scenario' : annotator_data['Scenario'][0],
            'fs_possible_actions' : annotator_data['Possible_actions'][0],
            'fs_selected_action': sel_act,
            'fs_action_type': action_criteria_list
        }

    return main_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3-8B', type=str, help='Model name for pipeline')       ## meta-llama/Meta-Llama-3-8B
    parser.add_argument('--language', default='English', type=str, help='Language out of ["English", "Chinese", "Russian", "Arabic", "Spanish", and "Hindi"]')
    parser.add_argument('--mode', default='desc', type=str, help='Mode out of ["desc", "moral", "culture", "fs", "np"]')
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling (default: 0)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for nucleus sampling (default: 0.95)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling (default: 50)")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Maximum number of tokens to generate (default: 32768)")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode with fewer samples")
    args = parser.parse_args()

    RQ = 2
    mode = args.mode
    language = args.language
    model_name = args.model

    with open("PROMPTS2.txt", "r") as f:
        PROMPTS = f.read()
    PROMPTS = ast.literal_eval(PROMPTS)

    model = utils.load_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data_file_rq2 = f"Final_data/{language}_long_formatted.csv"

    data = read_data_RQ2(data_file_rq2)

    print(data)
    formatted_prompts = []
    ground_truth = []
    predictions = []

    if args.debug:
        data = data[:2]
        print("Running in debug mode with 10 samples")

    for i, inst in tqdm(enumerate(data)):
        ground_truth.append(inst['gt'])

        possible_actions = ast.literal_eval(inst['actions'])
        if type(possible_actions[0]) != str:
            possible_actions = "(a) " + possible_actions[0][0] + "; (b) " + possible_actions[1][0]
        else:
            possible_actions = "(a) " + possible_actions[0] + "; (b) " + possible_actions[1]
        
        prompt = PROMPTS[f"prompt_{mode}_{language.lower()[:3]}_rq{RQ}"]

        prompt = prompt.replace("[SCENARIO]", inst['scenario'])
        prompt = prompt.replace("[ACTIONS]", possible_actions)
        if mode != 'fs':
            prompt = prompt.replace("[SELECTED_ACTION]", inst['action_selected'][0])
        else:
            prompt = prompt.replace("[SELECTED_ACTION]", inst['action_selected'])
        prompt = prompt.replace("[DESC]", inst['desc'])
        prompt = prompt.replace("[MORAL]", " ".join(map(str,inst['moral_vector'])))
        prompt = prompt.replace("[CULTURE]", " ".join(map(str,inst['culture_vector'])))

        prompt = prompt.replace("[MORAL_VALUE_1]", inst['moral_idx'][0])
        prompt = prompt.replace("[MORAL_VALUE_2]", inst['moral_idx'][1])
        prompt = prompt.replace("[MORAL_VALUE_3]", inst['moral_idx'][2])
        prompt = prompt.replace("[MORAL_VALUE_4]", inst['moral_idx'][3])
        prompt = prompt.replace("[MORAL_VALUE_5]", inst['moral_idx'][4])
        prompt = prompt.replace("[MORAL_VALUE_6]", inst['moral_idx'][5])
        
        prompt = prompt.replace("[CULTURAL_VALUE_1]", inst['culture_idx'][0])
        prompt = prompt.replace("[CULTURAL_VALUE_2]", inst['culture_idx'][1])
        prompt = prompt.replace("[CULTURAL_VALUE_3]", inst['culture_idx'][2])
        prompt = prompt.replace("[CULTURAL_VALUE_4]", inst['culture_idx'][3])
        prompt = prompt.replace("[CULTURAL_VALUE_5]", inst['culture_idx'][4])
        prompt = prompt.replace("[CULTURAL_VALUE_6]", inst['culture_idx'][5])

        prompt = prompt.replace("[FS_SCENARIO]", inst['fs_data']['fs_scenario'])
        prompt = prompt.replace("[FS_ACTION]", inst['fs_data']['fs_selected_action'])
        prompt = prompt.replace("[FS_ACTION_TYPE]", inst['fs_data']['fs_action_type'])

        formatted_prompt = formatted_prompt = f"""
            ### Instruction:
            {prompt}

            ### Response:
        """

        if i == 0:
            print("Example prompt: ", formatted_prompt)

        formatted_prompts.append(formatted_prompt)

    print("\nGenerating responses...")
    inputs, outputs = utils.get_response(model_name, model, tokenizer, formatted_prompts, 
                                         temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, 
                                         max_tokens=args.max_tokens)


    unsure_preds = 0
    predictions = []
    generations = []
    for output in outputs:
        generated_text = output.outputs[0].text if type(output) is not str else output

        generations.append(generated_text)

        try:
            generated_text = generated_text.split("Selected action is ")[1].strip()
        except:
            print("generated_text: ", generated_text)
            generated_text = generated_text
        response = generated_text.lower().strip()
        if 'deontological' in response:
            predictions.append(label2idx['Deontological'])
        elif 'utilitarianism' in response:
            predictions.append(label2idx['Utilitarianism'])
        elif 'rights-based' in response:
            predictions.append(label2idx['Rights-based'])
        elif 'virtuous' in response:
            predictions.append(label2idx['Virtuous'])
        else:
            unsure_preds += 1
            predictions.append(label2idx['Unsure'])

    ground_truth = [[label2idx[x] for x in y] for y in ground_truth]

    print(ground_truth, predictions)
    accuracy, precision, recall, f1 = calculate_metrics(ground_truth, predictions)

    results = {
        'formatted_prompts': formatted_prompts,
        'generations': generations,
        'predictions': predictions,
        'ground truth': ground_truth,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1 score': f1
    }

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    with open(f"Final_results/{language}_RQ{RQ}_{mode}_{model_name.split('/')[-1]}.json","w") as f:
        json.dump(results, f)