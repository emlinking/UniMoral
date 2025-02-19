# %%
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from bert_score import score
import pandas as pd
from tqdm import tqdm

# %%
from collections import defaultdict

def calculate_metrics_RQ23(true_values, pred_values):
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

# %%
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

# %%
result_path = "Final_results/"
all_results = {'RQ1': {}, 'RQ2': {}, 'RQ3': {}, 'RQ4': {}}

languages = ['Arabic', 'Chinese', 'English', 'Hindi', 'Russian', 'Spanish']
modes_rq1 = ['np', 'desc', 'moral2', 'culture2', 'fs']
modes_rq23 = ['np', 'desc', 'moral2', 'culture2', 'fs2']
modes_rq4 = ['long']
models = ['Phi-3.5-mini-instruct', 'Llama-3.1-8B-Instruct', 'DeepSeek-R1-Distill-Llama-8B']

rq123_columns = ['Language', 'NP-Phi', 'NP-LLAMA', 'NP-R1', 'Desc-Phi', 'Desc-LLAMA', 'Desc-R1', 'MV-Phi', 'MV-LLAMA', 'MV-R1', 'CV-Phi', 'CV-LLAMA', 'CV-R1', 'FS-Phi', 'FS-LLAMA', 'FS-R1']
rq123_columns = ['Language', 'NP-Phi-PS', 'NP-Phi-R', 'NP-LLAMA-PS', 'NP-LLAMA-R', 'NP-R1-PS', 'NP-R1-R', 'Desc-Phi-PS', 'Desc-Phi-R', 'Desc-LLAMA-PS', 'Desc-LLAMA-R', 'Desc-R1-PS', 'Desc-R1-R', 'MV-Phi-PS', 'MV-Phi-R', 'MV-LLAMA-PS', 'MV-LLAMA-R', 'MV-R1-PS', 'MV-R1-R', 'CV-Phi-PS', 'CV-Phi-R', 'CV-LLAMA-PS', 'CV-LLAMA-R', 'CV-R1-PS', 'CV-R1-R', 'FS-Phi-PS', 'FS-Phi-R', 'FS-LLAMA-PS', 'FS-LLAMA-R', 'FS-R1-PS', 'FS-R1-R']


rq1_results = pd.DataFrame(columns=rq123_columns)

reddit_cnt = {}
for lang in tqdm(languages):
    data_file_long = f"Final_data/{lang}_long_2.csv"
    data_file_short = f"Final_data/{lang}_short_2.csv"
    data = pd.read_csv(data_file_long)
    data_short = pd.read_csv(data_file_short)
    data = pd.concat([data, data_short], ignore_index=True)
    data = data[['Scenario_id']]
    print("Data read -- ", len(data), "\n", data.columns, "\n", data.head())
    
    row = [lang]
    for mode in modes_rq1:
        for model in models:
            file = f"{lang}_RQ1_{mode}_{model}.json"
            with open(os.path.join(result_path, file), 'r') as f:
                results = json.load(f)

            preds = results['predictions']
            gt = results['ground truth']
            scenario_id = data['Scenario_id'].tolist()

            ps_preds, ps_gt = [], []
            reddit_preds, reddit_gt = [], []

            for s_id, p, g in zip(scenario_id, preds, gt):
                if 'Reddit' in s_id:
                    reddit_preds.append(p)
                    reddit_gt.append(g)
                else:
                    ps_preds.append(p)
                    ps_gt.append(g)

            if lang not in reddit_cnt.keys():
                reddit_cnt[lang] = len(reddit_preds)

            # precision = precision_score(gt, preds, average='weighted')
            # recall = recall_score(gt, preds, average='weighted')
            # f1 = round(f1_score(gt, preds, average='weighted')*100, 2)
            reddit_f1 = round(f1_score(reddit_gt, reddit_preds, average='weighted')*100, 2)
            ps_f1 = round(f1_score(ps_gt, ps_preds, average='weighted')*100, 2)

            row.extend([ps_f1, reddit_f1])
    
    rq1_results.loc[len(rq1_results)] = row

rq1_results.to_csv('RQ1_results_Reddit.csv', index=False)
print(reddit_cnt)

rq123_columns = ['Language', 'NP-Phi', 'NP-LLAMA', 'NP-R1', 'Desc-Phi', 'Desc-LLAMA', 'Desc-R1', 'MV-Phi', 'MV-LLAMA', 'MV-R1', 'CV-Phi', 'CV-LLAMA', 'CV-R1', 'FS-Phi', 'FS-LLAMA', 'FS-R1']
rq2_results = pd.DataFrame(columns=rq123_columns)

reddit_cnt = {}
for lang in tqdm(languages):
    data_file_long = f"Final_data/{lang}_long_2.csv"
    data = pd.read_csv(data_file_long)
    data = data[['Scenario_id']]
    print("Data read -- ", len(data), "\n", data.columns, "\n", data.head())

    row = [lang]
    for mode in modes_rq23:
        for model in models:
            file = f"{lang}_RQ2_{mode}_{model}.json"
            with open(os.path.join(result_path, file), 'r') as f:
                results = json.load(f)

            preds = results['predictions']
            gt = results['ground truth']

            scenario_id = data['Scenario_id'].tolist()

            ps_preds, ps_gt = [], []
            reddit_preds, reddit_gt = [], []

            for s_id, p, g in zip(scenario_id, preds, gt):
                if 'Reddit' in s_id:
                    reddit_preds.append(p)
                    reddit_gt.append(g)
                else:
                    ps_preds.append(p)
                    ps_gt.append(g)

            if lang not in reddit_cnt.keys():
                reddit_cnt[lang] = len(reddit_preds)

            _,_,_, wtd_f1_reddit = calculate_metrics_RQ23(reddit_gt, reddit_preds)
            _,_,_, wtd_f1_ps = calculate_metrics_RQ23(ps_gt, ps_preds)
            wtd_f1_reddit = round(wtd_f1_reddit*100, 2)
            wtd_f1_ps = round(wtd_f1_ps*100, 2)

            row.extend([wtd_f1_ps, wtd_f1_reddit])

    rq2_results.loc[len(rq2_results)] = row

rq2_results.to_csv('RQ2_results_Reddit.csv', index=False)
print(reddit_cnt)

rq123_columns = ['Language', 'NP-Phi', 'NP-LLAMA', 'NP-R1', 'Desc-Phi', 'Desc-LLAMA', 'Desc-R1', 'MV-Phi', 'MV-LLAMA', 'MV-R1', 'CV-Phi', 'CV-LLAMA', 'CV-R1', 'FS-Phi', 'FS-LLAMA', 'FS-R1']
rq3_results = pd.DataFrame(columns=rq123_columns)

reddit_cnt = {}
for lang in tqdm(languages):
    data_file_long = f"Final_data/{lang}_long_2.csv"
    data = pd.read_csv(data_file_long)
    data = data[['Scenario_id']]
    print("Data read -- ", len(data), "\n", data.columns, "\n", data.head())

    row = [lang]
    for mode in modes_rq23:
        for model in models:
            file = f"{lang}_RQ3_{mode}_{model}.json"
            with open(os.path.join(result_path, file), 'r') as f:
                results = json.load(f)

            preds = results['predictions']
            gt = results['ground truth']
            scenario_id = data['Scenario_id'].tolist()

            ps_preds, ps_gt = [], []
            reddit_preds, reddit_gt = [], []

            for s_id, p, g in zip(scenario_id, preds, gt):
                if 'Reddit' in s_id:
                    reddit_preds.append(p)
                    reddit_gt.append(g)
                else:
                    ps_preds.append(p)
                    ps_gt.append(g)

            if lang not in reddit_cnt.keys():
                reddit_cnt[lang] = len(reddit_preds)

            _,_,_, wtd_f1_reddit = calculate_metrics_RQ23(reddit_gt, reddit_preds)
            _,_,_, wtd_f1_ps = calculate_metrics_RQ23(ps_gt, ps_preds)
            wtd_f1_reddit = round(wtd_f1_reddit*100, 2)
            wtd_f1_ps = round(wtd_f1_ps*100, 2)

            row.extend([wtd_f1_ps, wtd_f1_reddit])

            # precision, recall, f1, wtd_f1 = calculate_metrics_RQ23(gt, preds)
            # wtd_f1 = round(wtd_f1*100, 2)

            # row.append(wtd_f1)

    rq3_results.loc[len(rq3_results)] = row

rq3_results.to_csv('RQ3_results_Reddit.csv', index=False)
print(reddit_cnt)

rq4_columns = ['Language', 'BLEU-Phi-PS', 'METEOR-Phi-PS', 'BS-Phi-PS', 'BLEU-Phi-R', 'METEOR-Phi-R', 'BS-Phi-R', 'BLEU-LLAMA-PS', 'METEOR-LLAMA-PS', 'BS-LLAMA-PS', 'BLEU-LLAMA-R', 'METEOR-LLAMA-R', 'BS-LLAMA-R', 'BLEU-R1-PS', 'METEOR-R1-PS', 'BS-R1-PS', 'BLEU-R1-R', 'METEOR-R1-R', 'BS-R1-R']
rq4_results = pd.DataFrame(columns=rq4_columns)

reddit_cnt = {}
for lang in tqdm(languages):
    data_file_long = f"Final_data/{lang}_long_2.csv"
    data = pd.read_csv(data_file_long)
    data = data[['Scenario_id', 'Selected_action']]
    print("Data read -- ", len(data), "\n", data.columns, "\n", data.head())
    
    row = [lang]
    for mode in modes_rq4:
        for model in models:
            file = f"{lang}_RQ4_{mode}_{model}.json"
            with open(os.path.join(result_path, file), 'r') as f:
                results = json.load(f)

            preds = results['predictions']
            gt = results['ground truth']

            new_preds = []
            for pred in tqdm(preds):
                pred = " ".join(pred.lower().split()).strip()
                if 'consequence of the action is' in pred:
                    pred = pred.split('consequence of the action is')[1]
                new_preds.append(pred)

            scenario_id = []
            scenario_action = {}
            for s_id, act in zip(data['Scenario_id'], data['Selected_action']):
                if s_id not in scenario_action.keys():
                    scenario_action[s_id] = []
                if act not in scenario_action[s_id]:
                    scenario_action[s_id].append(act)
                    scenario_id.append(s_id)

            ps_preds, ps_gt = [], []
            reddit_preds, reddit_gt = [], []

            for s_id, p, g in zip(scenario_id, new_preds, gt):
                if 'Reddit' in s_id:
                    reddit_preds.append(p)
                    reddit_gt.append(g)
                else:
                    ps_preds.append(p)
                    ps_gt.append(g)

            if lang not in reddit_cnt.keys():
                reddit_cnt[lang] = len(reddit_preds)
            print(len(scenario_id), len(reddit_gt)+len(ps_gt))
            print("Calculating metrics...")
            scores = evaluate_metrics(reddit_preds, reddit_gt, bs_lang_dict[lang])
            reddit_bleu = round(scores['avg_bleu']*100, 2)
            reddit_meteor = round(scores['avg_meteor']*100, 2)
            reddit_bs = round(scores['avg_bert_score']*100, 2)

            scores = evaluate_metrics(ps_preds, ps_gt, bs_lang_dict[lang])
            ps_bleu = round(scores['avg_bleu']*100, 2)
            ps_meteor = round(scores['avg_meteor']*100, 2)
            ps_bs = round(scores['avg_bert_score']*100, 2)

            row.extend([ps_bleu, ps_meteor, ps_bs, reddit_bleu, reddit_meteor, reddit_bs])

    rq4_results.loc[len(rq4_results)] = row

rq4_results.to_csv('RQ4_results_reddit2.csv', index=False)
print(reddit_cnt)

# all_bs = []
# all_b = []
# all_m = []

# for lang in tqdm(languages):
#     data_file_long = f"Final_data/{lang}_long_2.csv"
#     data = pd.read_csv(data_file_long)
#     data = data[['Scenario_id', 'Selected_action']]
#     print("Data read -- ", len(data), "\n", data.columns, "\n", data.head())

#     scenario_id = []
#     scenario_action = {}
#     for s_id, act in zip(data['Scenario_id'], data['Selected_action']):
#         if s_id not in scenario_action.keys():
#             scenario_action[s_id] = []
#         if act not in scenario_action[s_id]:
#             scenario_action[s_id].append(act)
#             scenario_id.append(s_id)

#     row = [lang]
#     for mode in modes_rq4:
#         for model in models:
#             file = f"{lang}_RQ4_{mode}_{model}.json"
#             with open(os.path.join(result_path, file), 'r') as f:
#                 results = json.load(f)

#             preds = results['predictions']
#             gt = results['ground truth']

#             new_preds = []
#             for pred in tqdm(preds):
#                 pred = " ".join(pred.lower().split()).strip()
#                 if 'consequence of the action is' in pred:
#                     pred = pred.split('consequence of the action is')[1]
#                 new_preds.append(pred)

#             print(len(gt), len(preds), len(scenario_id))
#             min_len = min(len(gt), len(preds), len(scenario_id))
#             print(min_len)
#             df = pd.DataFrame({
#                 'true': gt[:min_len],
#                 'pred': new_preds[:min_len],
#                 's_id': scenario_id[:min_len]
#             })

#             random_preds = []
#             for i in range(len(df)):
#                 curr_s = df['s_id'][i]
#                 other_s = df[df['s_id'] != curr_s]
#                 p = other_s.sample(n=1, random_state=42)
#                 random_preds.append(p['pred'].values[0])

#             print(type(random_preds), type(random_preds[0]))
#             print("Calculating metrics...")
#             scores = evaluate_metrics(random_preds, gt, bs_lang_dict[lang])
#             bleu = round(scores['avg_bleu']*100, 2)
#             meteor = round(scores['avg_meteor']*100, 2)
#             bs = round(scores['avg_bert_score']*100, 2)
#             all_bs.append(bs)
#             all_b.append(bleu)
#             all_m.append(meteor)

# average_bs = sum(all_bs)/len(all_bs)
# average_b = sum(all_b)/len(all_b)
# average_m = sum(all_m)/len(all_m)
# print("\n\n\n\n\n")
# print(average_bs)
# print(average_b)
# print(average_m)
# # rq4_results.to_csv('RQ4_results_random.csv', index=False)
# # print(reddit_cnt)