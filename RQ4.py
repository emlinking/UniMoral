import os
os.environ['HF_HOME'] = "/shared/0/projects/code-switching/datasets"

from transformers import AutoTokenizer
from huggingface_hub import login
import argparse
import pandas as pd
import ast
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
import random
import numpy as np
import torch
import nltk
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from bert_score import score
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
bs_lang_dict = {'English': 'en', 'Spanish': 'es', 'Arabic': 'ar', 'Russian': 'ru', 'Chinese': 'zh', 'Hindi': 'hi'}


def evaluate_metrics(predictions, ground_truths, bs_lang):
    smoothing_function = SmoothingFunction().method1
    
    bleu_scores = []
    meteor_scores = []
    bert_f1_scores = []
    
    for pred, refs in zip(predictions, ground_truths):
        sample_bleu = []
        pred_tokens = pred.split()
        for ref in refs:
            ref_tokens = ref.split()
            score_value = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                smoothing_function=smoothing_function
            )
            sample_bleu.append(score_value)
        max_bleu = max(sample_bleu) if sample_bleu else 0.0
        bleu_scores.append(max_bleu)

        sample_meteor = []
        for ref in refs:
            score_value = single_meteor_score(ref.split(), pred.split())
            sample_meteor.append(score_value)
        max_meteor = max(sample_meteor) if sample_meteor else 0.0
        meteor_scores.append(max_meteor)
        
        sample_bert = []
        for ref in refs:
            P, R, F1 = score([pred], [ref], lang=bs_lang, verbose=False)
            sample_bert.append(F1.item())
        max_bert = max(sample_bert) if sample_bert else 0.0
        bert_f1_scores.append(max_bert)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    avg_bert = sum(bert_f1_scores) / len(bert_f1_scores) if bert_f1_scores else 0.0

    return {
        'bleu': bleu_scores,
        'avg_bleu': avg_bleu,
        'meteor': meteor_scores,
        'avg_meteor': avg_meteor,
        'bert_score': bert_f1_scores,
        'avg_bert_score': avg_bert
    }

def dict_to_list_of_vectors(data):
    data = ast.literal_eval(data)
    return list(data.values())

def read_data_RQ4(data_file_long):
    print("Reading data...")
    data = pd.read_csv(data_file_long)
    data = data[['Scenario_id', 'Annotator_id', 'Scenario', 'Possible_actions', 'Selected_action', 'Consequence', 'Moral_values', 'Cultural_values', 'Annotator_self_description']]
    print("Data read -- ", len(data), "\n", data.columns, "\n", data.head())

    scenario_consequences = {}

    unique_scenario_id = data['Scenario_id'].unique()
    for s_id in unique_scenario_id:
        s_data = data[data['Scenario_id'] == s_id]
        action_consequence = {}
        for acts, sel_act, con in zip(s_data['Possible_actions'], s_data['Selected_action'], s_data['Consequence']):
            if type(ast.literal_eval(acts)[0]) != str:
                sel_act = act2idx[sel_act] + ": " + ast.literal_eval(acts)[0][sel_act-1]
            else:
                sel_act = act2idx[sel_act] + ": " + ast.literal_eval(acts)[sel_act-1]

            try:
                if math.isnan(float(con)):
                    continue
            except:
                con = con

            con = con.replace('[','')
            con = con.replace(']','')
            con = " ".join(con.lower().strip().split())

            if sel_act not in action_consequence.keys():
                action_consequence[sel_act] = []
            action_consequence[sel_act].append(con)
        scenario_consequences[s_id] = action_consequence

    main_data = []
    for s_id, act_con in scenario_consequences.items():
        scenario = data[data['Scenario_id'] == s_id]['Scenario'].values[0]
        for sel_act, cons in act_con.items():
            this_inst = {
                'scenario_id': s_id,
                'scenario': scenario,
                'selected_action': sel_act,
                'gt': cons
            }
            main_data.append(this_inst)

    return main_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3-8B', type=str, help='Model name for pipeline')       ## meta-llama/Meta-Llama-3-8B
    parser.add_argument('--language', default='English', type=str, help='Language out of ["English", "Chinese", "Russian", "Arabic", "Spanish", and "Hindi"]')
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling (default: 0)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for nucleus sampling (default: 0.95)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling (default: 50)")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Maximum number of tokens to generate (default: 32768)")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode with fewer samples")
    parser.add_argument("--utilization", type=float, default=0.9, help="GPU memory utilization for model loading (default: 0.9)")
    args = parser.parse_args()

    RQ = 4
    language = args.language
    bs_lang = bs_lang_dict[language]
    model_name = args.model
    batch_size = args.batch_size

    with open("PROMPTS4.txt", "r") as f:
        PROMPTS = f.read()
    PROMPTS = ast.literal_eval(PROMPTS)

    model = utils.load_model(model_name, utilization=args.utilization)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data_file_rq4 = f"Final_data/{language}_long_formatted.csv"
    data = read_data_RQ4(data_file_rq4)

    formatted_prompts = []
    ground_truth = []
    predictions = []

    if args.debug:
        data = data[:2]
        print("Running in debug mode with 2 samples")

    for i, inst in tqdm(enumerate(data)):
        ground_truth.append(inst['gt'])

        prompt = PROMPTS[f"prompt_{language.lower()[:3]}_rq{RQ}"]
        
        prompt = prompt.replace("[SCENARIO]", inst['scenario'])
        prompt = prompt.replace("[SELECTED_ACTION]", inst['selected_action'])

        if "instruct" in model_name:
            formatted_prompt = formatted_prompt = f"""
                ### Instruction:
                {prompt}

                ### Response:
            """
        else:
            formatted_prompt = [
                {"role": "user", "content": prompt},
            ]

        if i == 0:
            print("Example prompt: ", formatted_prompt)

        formatted_prompts.append(formatted_prompt)

    print("\nGenerating responses...")
    inputs, outputs = utils.get_response(model_name, model, tokenizer, formatted_prompts, 
                                         temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, 
                                         max_tokens=args.max_tokens)

    predictions = []
    generations = []
    for output in outputs:
        generated_text = output.outputs[0].text if type(output) is not str else output
        generations.append(generated_text)

        try:
            generated_text = generated_text.split("consequence of the action is")[1].strip()
        except:
            print("generated_text: ", generated_text)
            generated_text = generated_text
 
        predictions.append(generated_text)

    print(ground_truth, predictions)
    results = evaluate_metrics(predictions, ground_truth, bs_lang)
    print("Average BLEU Score:", results['avg_bleu'])
    print("Average METEOR Score:", results['avg_meteor'])
    print("Average BERTScore (F1):", results['avg_bert_score'])

    results = {
        'formatted_prompts': formatted_prompts,
        'generations': generations,
        'predictions': predictions,
        'ground truth': ground_truth,
        'Per sample BLEU': results['bleu'],
        'Average BLEU': results['avg_bleu'],
        'Per sample METEOR': results['meteor'],
        'Average METEOR': results['avg_meteor'],
        'Per sample BERTScore': results['bert_score'],
        'Average BERTScore': results['avg_bert_score'],
    }

    with open(f"Final_results/{language}_RQ{RQ}_long_{model_name.split('/')[-1]}.json","w") as f:
        json.dump(results, f, ensure_ascii=False)