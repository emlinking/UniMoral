# Run the file: for BART -- nohup python -u finetune.py --model_type bart --summary_type general > OUTs/BART_FT_general.out &
# Run the file: for T5 -- nohup python -u finetune.py --model_type t5 --summary_type pd > OUTs/T5_FT_pd.out &

import os
import argparse
import numpy as np
import pandas as pd
import json
import warnings
import logging
import random
import math
import re
from tqdm import tqdm
from datetime import datetime

from collections import Counter

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers.modeling_utils import PreTrainedModel, unwrap_model

from transformers import (
    AutoTokenizer,
    T5TokenizerFast,
    LlamaModel,
    AdamW
)

from huggingface_hub import login
access_token = "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"
login(token = access_token)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:2")
    print("Using GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# DEVICE = torch.device("cpu")

# -------------------------------------------------------------- CONFIG -------------------------------------------------------------- #

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = 42
set_random_seed(SEED)

TARGET_MAX_LEN = 1

BATCH_SIZE = 4
MAX_EPOCHS = 10

EARLY_STOPPING = True
EARLY_STOPPING_THRESHOLD = 2

SCALER = GradScaler()

input_label_sets = ['anecdotes', 'action_classification', 'consequence_classification', 'commonsense', 'deontology', 'justice', 'virtue', 'moralexceptqa', 'moralfoundationredditcorpus', 'storycommonsense', 'moralconvita', 'moralfoundationstwittercorpus']
input_labels_sets = ['moralintegritycorpus']
inputs_label_sets = ['dilemmas', 'utilitarianism', 'storal_en', 'storal_zh']

# ------------------------------------------------------------- DATA UTILS -------------------------------------------------------------- #

def read_csv_data(path):
    data = pd.read_csv(path)
    if 'labels' in data.columns:
        data.rename(columns = {'labels':'label'}, inplace = True)
    return data

def pad_seq(tensor: torch.tensor, dim: int, max_len: int):
    return torch.cat([tensor, torch.zeros(max_len - tensor.shape[0], dim)])

def get_multilabel_ready(label):
    lab = str(label)
    lab = re.sub("\[","", label)
    lab = re.sub("\]","", lab)
    lab = re.sub("\"","", lab)
    lab = re.sub("\'","", lab)
    lab = [x.strip() for x in lab.split(",")]
    return lab

def preprocess_dataset(DATASET, text_path: str):
    dataset = read_csv_data(text_path)

    if DATASET in input_label_sets:
        source = [SOURCE_PREFIX + str(s)for s in dataset['input'].tolist()]
        model_inputs = TOKENIZER(source, max_length=SOURCE_MAX_LEN, padding='max_length', truncation=True)
        
        all_labels = dataset['label'].tolist()
        idx2lab = list(set(all_labels))
        lab2idx = {k:v for v,k in enumerate(idx2lab)}

        wts = dict(Counter(all_labels))
        weights = torch.zeros((len(idx2lab)), dtype=torch.long).to(DEVICE)
        for k,v in wts.items():
            weights[lab2idx[k]] = v/max(wts.values())
        print("weights: ", weights)

        target = [lab2idx[t] for t in dataset['label'].tolist()]

        model_inputs['input_ids'] = torch.tensor([i for i in model_inputs['input_ids']], dtype=torch.long, device=DEVICE)
        model_inputs['attention_mask'] = torch.tensor([a for a in model_inputs['attention_mask']], dtype=torch.long, device=DEVICE)
        model_inputs['labels'] = torch.tensor(target, dtype=torch.long, device=DEVICE)

    elif DATASET in input_labels_sets:
        source = [SOURCE_PREFIX + s for s in dataset['input'].tolist()]
        model_inputs = TOKENIZER(source, max_length=SOURCE_MAX_LEN, padding='max_length', truncation=True)

        all_labels = [get_multilabel_ready(x) for x in dataset['label']]
        idx2lab = list(set([x for xs in all_labels for x in xs]))
        lab2idx = {k:v for v,k in enumerate(idx2lab)}

        wts = dict(Counter(all_labels))
        weights = torch.zeros((len(idx2lab)), dtype=torch.long).to(DEVICE)
        for k,v in wts.items():
            weights[lab2idx[k]] = v/max(wts.values())
        print("weights: ", weights)

        target = MultiLabelBinarizer().fit_transform(all_labels)

        model_inputs['input_ids'] = torch.tensor([i for i in model_inputs['input_ids']], dtype=torch.long, device=DEVICE)
        model_inputs['attention_mask'] = torch.tensor([a for a in model_inputs['attention_mask']], dtype=torch.long, device=DEVICE)
        model_inputs['labels'] = torch.tensor(target, dtype=torch.long, device=DEVICE)

    elif DATASET in inputs_label_sets:
        all_labels = dataset['label'].tolist()
        idx2lab = list(set(all_labels))
        lab2idx = {k:v for v,k in enumerate(idx2lab)}

        wts = dict(Counter(all_labels))
        print("wts: ", wts)
        weights = torch.zeros((len(idx2lab)), dtype=torch.long).to(DEVICE)
        for k,v in wts.items():
            weights[lab2idx[k]] = v/max(wts.values())
        print("weights: ", weights)

        target = [lab2idx[t] for t in dataset['label'].tolist()]

        sources_input_ids = []
        sources_attn_mask = []
        inp_cols = [x for x in dataset.columns if x != 'label' and "Unnamed" not in x]
        for inp in inp_cols:
            source = [SOURCE_PREFIX + s for s in dataset[inp].tolist()]
            source = TOKENIZER(source, max_length=SOURCE_MAX_LEN, padding='max_length', truncation=True)
            source['input_ids'] = torch.tensor([i for i in source['input_ids']], dtype=torch.long, device=DEVICE)
            source['attention_mask'] = torch.tensor([a for a in source['attention_mask']], dtype=torch.long, device=DEVICE)
            sources_input_ids.append(source['input_ids'])
            sources_attn_mask.append(source['attention_mask'])
        
        model_inputs = {
            'input_ids': torch.stack(sources_input_ids, dim=1),
            'attention_mask': torch.stack(sources_attn_mask,dim=1),
            'labels': torch.tensor(target, dtype=torch.long, device=DEVICE)
        }

    return model_inputs, lab2idx

def set_up_data_loader(dataset, text_path: str, set_type: str):
    dataset, lab2idx = preprocess_dataset(DATASET=dataset, text_path=text_path)
    dataset = TensorDataset(dataset['input_ids'], dataset['attention_mask'], dataset['labels'])

    if set_type == 'test':         ## No shuffling for test set
        return lab2idx, DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return lab2idx, DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def pad_to_max_len(tokenizer, tensor, max_length):
        if tokenizer is None:
            raise ValueError(
                f"Tensor need to be padded to `max_length={max_length}` but no tokenizer was passed when creating "
                "this `Trainer`. Make sure to create your `Trainer` with the appropriate tokenizer."
            )
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        padded_tensor = pad_token_id * torch.ones((tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device)
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

def get_scores(reference_list: list, hypothesis_list: list):
    acc = accuracy_score(reference_list, hypothesis_list)
    report = classification_report(reference_list, hypothesis_list)

    return {"accuracy": acc, "report": report}

def _save(model, output_dir: str, tokenizer=None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model, PreTrainedModel):
            if isinstance(unwrap_model(model), PreTrainedModel):
                if state_dict is None:
                    state_dict = model.state_dict()
                unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, DATASET1 + "_" + DATASET2))
        else:
            model.save_pretrained(output_dir, state_dict=state_dict)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

def save_model(model, output_dir: str, tokenizer=None, state_dict=None):
        _save(model,output_dir, tokenizer=tokenizer, state_dict=state_dict)

def load_model(model, input_dir: str):
    state_dict = torch.load(os.path.join(input_dir, DATASET1 + "_" + DATASET2))
    model.load_state_dict(state_dict)
    return model

# ----------------------------------------------------- MODEL ----------------------------------------------------- #

class MyLlamaModelMultiTask(nn.Module):
    def __init__(self, pretrained_model, num_classes_T1 = 2, num_classes_T2 = 2):
        super().__init__()
        self.encoder = LlamaModel.from_pretrained(pretrained_model)
        
        for (name, param) in self.encoder.named_parameters():
            if "layers.31" not in name and "layers.30" not in name and "layers.29" not in name and "layers.28" not in name and name != "norm.weight":
                param.requires_grad = False
        for (name, param) in self.encoder.named_parameters():
            print(name, param.requires_grad)

        self.task1_head = nn.Sequential(
            nn.LayerNorm(4096),
            nn.Linear(4096, 768), 
            nn.LayerNorm(768),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(768, num_classes_T1)
        )

        self.task2_head = nn.Sequential(
            nn.LayerNorm(4096),
            nn.Linear(4096, 768), 
            nn.LayerNorm(768),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(768, num_classes_T2)
        )
        
    def forward(self, source1=[], source2=[]):
        op1, op2 = [], []
        if source1:
            if DATASET1 in input_label_sets:
                op = self.encoder(source1['input_ids'], source1['attention_mask']).last_hidden_state.mean(dim=1)
                op1 = self.task1_head(op)

            elif DATASET1 in inputs_label_sets:
                op = []
                source1['input_ids'] = torch.transpose(source1['input_ids'], 0, 1)
                source1['attention_mask'] = torch.transpose(source1['attention_mask'], 0, 1)
                for input_ids, attention_mask in zip(source1['input_ids'], source1['attention_mask']):
                    op.append(self.encoder(input_ids, attention_mask).last_hidden_state.mean(dim=1))
                op = torch.stack(op).mean(dim=0)
                op1 = self.task1_head(op)

            elif DATASET1 in input_labels_sets:
                op = self.encoder(source1['input_ids'], source1['attention_mask']).last_hidden_state.mean(dim=1)
                op1 = self.task1_head(op)

        if source2:
            if DATASET2 in input_label_sets:
                op = self.encoder(source2['input_ids'], source2['attention_mask']).last_hidden_state.mean(dim=1)
                op2 = self.task2_head(op)

            elif DATASET2 in inputs_label_sets:
                op = []
                source2['input_ids'] = torch.transpose(source2['input_ids'], 0, 1)
                source2['attention_mask'] = torch.transpose(source2['attention_mask'], 0, 1)
                for input_ids, attention_mask in zip(source2['input_ids'], source2['attention_mask']):
                    op.append(self.encoder(input_ids, attention_mask).last_hidden_state.mean(dim=1))
                op = torch.stack(op).mean(dim=0)
                op2 = self.task2_head(op)

            elif DATASET2 in input_labels_sets:
                op = self.encoder(source2['input_ids'], source2['attention_mask']).last_hidden_state.mean(dim=1)
                op2 = self.task2_head(op)

        if type(op1) == torch.Tensor and type(op2) == torch.Tensor:
            return op1, op2
        elif type(op1) == torch.Tensor:
            return op1
        elif type(op2) == torch.Tensor:
            return op2
    
# ----------------------------------------------------- TRAINING UTILS ----------------------------------------------------- #

def prepare_for_training(model, learning_rate):
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    return optimizer, criterion1, criterion2

def train_epoch(model, data_loader1, data_loader2, optimizer, criterion1, criterion2):
    model.train()
    epoch_train_loss = 0.0
    for step, (batch1, batch2) in enumerate(tqdm(zip(data_loader1, data_loader2), desc="Training Iteration")):
        input_ids, attention_mask, target1 = batch1
        source1 = {
            'input_ids': input_ids.to(DEVICE),
            'attention_mask': attention_mask.to(DEVICE)
        }

        input_ids, attention_mask, target2 = batch2
        source2 = {
            'input_ids': input_ids.to(DEVICE),
            'attention_mask': attention_mask.to(DEVICE)
        }

        target1 = target1.to(DEVICE)
        target2 = target2.to(DEVICE)
        optimizer.zero_grad()

        logits1, logits2 = model(source1=source1,  source2=source2)

        loss1 = criterion1(logits1, target1)
        loss2 = criterion2(logits2, target2)

        loss = (loss1 + loss2)/2
        epoch_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        # if step == 2:
        #     break

    torch.cuda.empty_cache()
    
    return epoch_train_loss/ step

def val_epoch(model, data_loader1, data_loader2, criterion1, criterion2):
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for step, (batch1, batch2) in enumerate(tqdm(zip(data_loader1,data_loader2), desc="Validation Iteration")):
            input_ids, attention_mask, target1 = batch1
            source1 = {
                'input_ids': input_ids.to(DEVICE),
                'attention_mask': attention_mask.to(DEVICE)
            }

            input_ids, attention_mask, target2 = batch2
            source2 = {
                'input_ids': input_ids.to(DEVICE),
                'attention_mask': attention_mask.to(DEVICE)
            }

            target1 = target1.to(DEVICE)
            target2 = target2.to(DEVICE)

            logits1, logits2 = model(source1, source2)

            loss1 = criterion1(logits1, target1)
            loss2 = criterion2(logits2, target2)

            loss = (loss1 + loss2)/2
            epoch_val_loss += loss.item()  
            # if step == 2:
            #     break

    torch.cuda.empty_cache() 
    
    return epoch_val_loss/ step

def test_epoch(model, data_loader1, data_loader2, desc):
    model.eval()
    predictions1, predictions2 = [], []
    gold1, gold2 = [], []
    with torch.no_grad():
        for step, batch1 in enumerate(tqdm(data_loader1, desc=desc)):
            input_ids, attention_mask, target1 = batch1
            source1 = {
                'input_ids': input_ids.to(DEVICE),
                'attention_mask': attention_mask.to(DEVICE)
            }

            target1 = target1.to(DEVICE)
            logits1 = model(source1 = source1)

            predicted_classes1 = torch.argmax(nn.Softmax(dim=1)(logits1), dim=1)
            predictions1.extend(predicted_classes1.tolist())
            gold1.extend(target1.tolist())

            # if step == 2:
            #     break

        for step, batch2 in enumerate(tqdm(data_loader2, desc=desc)):
            input_ids, attention_mask, target2 = batch2
            source2 = {
                'input_ids': input_ids.to(DEVICE),
                'attention_mask': attention_mask.to(DEVICE)
            }

            target2 = target2.to(DEVICE)
            logits2 = model(source2 = source2)

            predicted_classes2 = torch.argmax(nn.Softmax(dim=1)(logits2), dim=1)
            predictions2.extend(predicted_classes2.tolist())
            gold2.extend(target2.tolist())
            
            # if step == 2:
            #     break

    torch.cuda.empty_cache()
    return predictions1, gold1, predictions2, gold2

def compute_metrics(model, data_loader1, data_loader2, desc, **gen_kwargs):

    predictions1, gold1, predictions2, gold2 = test_epoch(model, data_loader1, data_loader2, desc=desc)
    result1 = get_scores(gold1, predictions1)
    result2 = get_scores(gold2, predictions2)

    torch.cuda.empty_cache() 
    
    return predictions1, gold1, predictions2, gold2, result1, result2

def train(model, tokenizer, train_data_loader1, train_data_loader2, val_data_loader1, val_data_loader2, test_data_loader1, test_data_loader2, learning_rate, model_type, **gen_kwargs):
    train_losses = []
    val_losses = []
    
    optimizer, criterion1, criterion2 = prepare_for_training(model=model, learning_rate=learning_rate)
    
    min_val_loss = 99999
    bad_loss = 0
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_data_loader1, train_data_loader2, optimizer, criterion1, criterion2)
        train_losses.append(train_loss)
        print("Epoch: {}\ttrain_loss: {}".format(epoch+1, train_loss))

        val_loss = val_epoch(model, val_data_loader1, val_data_loader2, criterion1, criterion2)
        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            bad_loss = 0
            min_val_loss = val_loss
            test_pred1, test_gold1, test_pred2, test_gold2, test_results1, test_results2 = compute_metrics(model, test_data_loader1, test_data_loader2, desc="Test Iteration", **gen_kwargs)
            
            print("Epoch: {}\ttrain_loss: {}\tval_loss: {}\tmin_val_loss: {}".format(epoch+1, train_loss, val_loss, min(val_losses)))
            print("\nTest Accuracy for DATASET ", DATASET1, ": ", test_results1['accuracy'])
            print("\nTest Classification Report for DATASET ", DATASET1, ":\n", test_results1['report'])

            print("\nTest Accuracy for DATASET ", DATASET2, ": ", test_results2['accuracy'])
            print("\nTest Classification Report for DATASET ", DATASET2, ":\n", test_results2['report'])
        
            path = OUTPUT_DIR + model_type
            save_model(model, path, tokenizer)
            print("Model saved at path: ", path)

            # test_df = pd.DataFrame()
            # test_df['Predictions'] = test_pred
            # test_df['Gold'] = test_gold

            # csv_path = OUTPUT_DIR + model_type + "_"  + TARGET_COLUMN + '.csv'
            # test_df.to_csv(csv_path)
            # print("Generations saved at path: ", path)
        else:
            bad_loss += 1
        
        if bad_loss == EARLY_STOPPING_THRESHOLD:
            print("Stopping early...")
            break

        torch.cuda.empty_cache()

# ------------------------------------------------------------ MAIN MODEL ------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='llama-3', type=str, help='Choose from [llama-3]')
    parser.add_argument('--file_pathT1', default='./', type=str, help='The path containing train/val/test.csv for Task 1')
    parser.add_argument('--file_pathT2', default='./', type=str, help='The path containing train/val/test.csv for Task 2')
    args = parser.parse_args()

    TEXT_INPUT_PATH_T1 = "/".join(args.file_pathT1.split("/")[:-1])
    TEXT_INPUT_PATH_T2 = "/".join(args.file_pathT2.split("/")[:-1])
    DATASET1 = args.file_pathT1.split("/")[-2]
    DATASET2 = args.file_pathT2.split("/")[-2]
    
    if args.model_type == 'llama_3':
        SOURCE_MAX_LEN = 1024
        SOURCE_PREFIX = ''
        print("Using Llama 3")
        TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", truncation_side='left')
        TOKENIZER.pad_token = TOKENIZER.eos_token
        print("Llama 3 Tokenizer loaded...\n")
    elif args.model_type == 'flan-t5':
        print("Using Flan-T5")
        TOKENIZER = T5TokenizerFast.from_pretrained('google/flan-t5-base', truncation_side='left')
        print("Flan-T5 Tokenizer loaded...\n")
        SOURCE_MAX_LEN = 1024
        SOURCE_PREFIX = 'Classify: '
    else:
        print("Error: Wrong model type")
        exit(0)
    # ------------------------------ READ DATASET ------------------------------ #
    
    lab2idx1, train_dataset1 = set_up_data_loader(dataset=DATASET1, text_path = TEXT_INPUT_PATH_T1 + '/train.csv', set_type = 'train')
    if "val.csv" in os.listdir(args.file_pathT1):
        _, val_dataset1 = set_up_data_loader(dataset=DATASET1, text_path = TEXT_INPUT_PATH_T1 + '/val.csv', set_type = 'val')
    else:
        _, val_dataset1 = set_up_data_loader(dataset=DATASET1, text_path = TEXT_INPUT_PATH_T1 + '/test.csv', set_type = 'val')
    _, test_dataset1 = set_up_data_loader(dataset=DATASET1, text_path = TEXT_INPUT_PATH_T1+ '/test.csv', set_type = 'test')

    lab2idx2, train_dataset2 = set_up_data_loader(dataset=DATASET2, text_path = TEXT_INPUT_PATH_T2 + '/train.csv', set_type = 'train')
    if "val.csv" in os.listdir(args.file_pathT2):
        _, val_dataset2 = set_up_data_loader(dataset=DATASET2, text_path = TEXT_INPUT_PATH_T2 + '/val.csv', set_type = 'val')
    else:
        _, val_dataset2 = set_up_data_loader(dataset=DATASET2, text_path = TEXT_INPUT_PATH_T2 + '/test.csv', set_type = 'val')
    _, test_dataset2 = set_up_data_loader(dataset=DATASET2, text_path = TEXT_INPUT_PATH_T2+ '/test.csv', set_type = 'test')
    
    print(len(train_dataset1), len(train_dataset2))
    # ------------------------------ MODEL SETUP ------------------------------ #
        
    if args.model_type == 'llama_3':
        MODEL = MyLlamaModelMultiTask("meta-llama/Meta-Llama-3-8B", num_classes_T1=len(lab2idx1.keys()), num_classes_T2=len(lab2idx2.keys()))
        print("Llama Model loaded...\n")
        print(MODEL)
        OUTPUT_DIR = "./models/llama_3/multitask/"
        # MODEL = load_model(MODEL, OUTPUT_DIR)
        # print("Model State Dict Loaded")
        LEARNING_RATE = 5e-6
    
    else:
        print("Error: Wrong model type")
        exit(0)

    MODEL.to(DEVICE)
    print(LEARNING_RATE)
    print(OUTPUT_DIR)
    print(SOURCE_PREFIX)

    gen_kwargs = {
        'early_stopping': EARLY_STOPPING,
    }
    
    # ------------------------------ TRAINING SETUP ------------------------------ #
    train(model=MODEL,
          tokenizer=TOKENIZER,
          train_data_loader1=train_dataset1,
          train_data_loader2=train_dataset2,
          val_data_loader1=val_dataset1,
          val_data_loader2=val_dataset2,
          test_data_loader1=test_dataset1,
          test_data_loader2=test_dataset2,
          learning_rate=LEARNING_RATE,
          model_type=args.model_type,
          **gen_kwargs)
    
    print("Model Trained!")