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
    RobertaTokenizerFast,
    T5TokenizerFast,
    RobertaModel,
    T5EncoderModel,
    AdamW
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
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
EARLY_STOPPING_THRESHOLD = 20

SCALER = GradScaler()

input_label_sets = ['anecdotes', 'action_classification', 'consequence_classification', 'commonsense', 'deontology', 'justice', 'virtue', 'moralexceptqa']
input_labels_sets = ['moralfoundationredditcorpus', 'storycommonsense', 'moralconvita', 'moralfoundationstwittercorpus', 'moralintegritycorpus']
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

def preprocess_dataset(text_path: str):
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
            weights[lab2idx[k]] = max(wts.values())/v
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
            weights[lab2idx[k]] = max(wts.values())/v
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
            weights[lab2idx[k]] = float(max(wts.values())/v)
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

def set_up_data_loader(text_path: str, set_type: str):
    dataset, lab2idx = preprocess_dataset(text_path=text_path)
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
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model.save_pretrained(output_dir, state_dict=state_dict)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

def save_model(model, output_dir: str, tokenizer=None, state_dict=None):
        _save(model,output_dir, tokenizer=tokenizer, state_dict=state_dict)

# ----------------------------------------------------- MODEL ----------------------------------------------------- #

class RoBERTaModel(nn.Module):
    def __init__(self, pretrained_model, num_classes = 2):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(pretrained_model)
        self.linear1 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.linear2 = nn.Linear(768, num_classes)
        
    def forward(self, source):
        # print("source['input_ids'] = ", source['input_ids'].size())
        # print("source['attention_mask'] = ", source['attention_mask'].size())
        if DATASET in input_label_sets:
            op = self.encoder(source['input_ids'], source['attention_mask']).pooler_output
            op = self.linear2(self.dropout(self.linear1(op)))

        elif DATASET in inputs_label_sets:
            op = []
            source['input_ids'] = torch.transpose(source['input_ids'], 0, 1)
            source['attention_mask'] = torch.transpose(source['attention_mask'], 0, 1)
            for input_ids, attention_mask in zip(source['input_ids'], source['attention_mask']):
                op.append(self.encoder(input_ids, attention_mask).pooler_output)
            op = torch.stack(op).mean(dim=0)
            op = self.linear2(self.dropout(self.linear1(op)))

        elif DATASET in input_labels_sets:
            op = self.encoder(source['input_ids'], source['attention_mask']).pooler_output
            op = self.linear2(self.dropout(self.linear1(op)))

        return op
    
class T5Model(nn.Module):
    def __init__(self, pretrained_model, num_classes = 2):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(pretrained_model)
        self.linear1 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.linear2 = nn.Linear(768, num_classes)
        
    def forward(self, source):
        if DATASET in input_label_sets:
            op = self.encoder(source['input_ids'], source['attention_mask']).last_hidden_state.mean(dim=1)
            op = self.linear2(self.dropout(self.linear1(op)))

        elif DATASET in inputs_label_sets:
            op = []
            for input_ids, attention_mask in zip(source['input_ids'], source['attention_mask']):
                op.append(self.encoder(input_ids, attention_mask).last_hidden_state.mean(dim=1).unsqueeze(1))
            op = torch.cat(op, dim = 1).mean(dim = 1)
            op = self.linear2(self.dropout(self.linear1(op)))

        elif DATASET in input_labels_sets:
            op = self.encoder(source['input_ids'], source['attention_mask']).last_hidden_state.mean(dim=1)
            op = self.linear2(self.dropout(self.linear1(op)))

        return op
# ----------------------------------------------------- TRAINING UTILS ----------------------------------------------------- #

def prepare_for_training(model, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    return optimizer, criterion

def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    epoch_train_loss = 0.0
    for step, batch in enumerate(tqdm(data_loader, desc="Training Iteration")):
        input_ids, attention_mask, target = batch
        source = {
            'input_ids': input_ids.to(DEVICE),
            'attention_mask': attention_mask.to(DEVICE)
        }

        target = target.to(DEVICE)
        optimizer.zero_grad()
        logits = model(source)
        loss = criterion(logits, target)
        epoch_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        # break

    torch.cuda.empty_cache()
    
    return epoch_train_loss/ step

def val_epoch(model, data_loader, criterion):
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc="Validation Iteration")):
            input_ids, attention_mask, target = batch
            source = {
                'input_ids': input_ids.to(DEVICE),
                'attention_mask': attention_mask.to(DEVICE)
            }
            target = target.to(DEVICE)

            logits = model(source)
            loss = criterion(logits, target)
            epoch_val_loss += loss.item()  

    torch.cuda.empty_cache() 
    
    return epoch_val_loss/ step

def test_epoch(model, data_loader, desc):
    model.eval()
    predictions = []
    gold = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc=desc)):
            input_ids, attention_mask, target = batch
            source = {
                'input_ids': input_ids.to(DEVICE),
                'attention_mask': attention_mask.to(DEVICE)
            }

            target = target.to(DEVICE)
            logits = model(source)

            predicted_classes = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)

            predictions.extend(predicted_classes.tolist())
            gold.extend(target.tolist())

    torch.cuda.empty_cache()
    return predictions, gold

def compute_metrics(model, tokenizer,data_loader, desc, **gen_kwargs):

    predictions, gold = test_epoch(model, data_loader, desc=desc)
    result = get_scores(gold, predictions)

    torch.cuda.empty_cache() 
    
    return predictions, gold, result  

def train(model, tokenizer, train_data_loader, val_data_loader, test_data_loader, learning_rate, model_type, **gen_kwargs):
    train_losses = []
    val_losses = []
    
    optimizer, criterion = prepare_for_training(model=model, learning_rate=learning_rate)
    
    patience = 0
    min_val_loss = 99999
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_data_loader, optimizer, criterion)
        train_losses.append(train_loss)
        print("Epoch: {}\ttrain_loss: {}".format(epoch+1, train_loss))

        val_loss = val_epoch(model, val_data_loader, criterion)
        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            test_pred, test_gold, test_results = compute_metrics(model, tokenizer, test_data_loader, desc="Test Iteration", **gen_kwargs)
            
            print("Epoch: {}\ttrain_loss: {}\tval_loss: {}\tmin_val_loss: {}".format(epoch+1, train_loss, val_loss, min(val_losses)))
            print("\nTest Accuracy: ", test_results['accuracy'])
            print("\nTest Classification Report:\n", test_results['report'])
        
            # path = OUTPUT_DIR + model_type + "_"  + TARGET_COLUMN
            # save_model(model, path, tokenizer)
            # print("Model saved at path: ", path)

            # test_df = pd.DataFrame()
            # test_df['Predictions'] = test_pred
            # test_df['Gold'] = test_gold

            # csv_path = OUTPUT_DIR + model_type + "_"  + TARGET_COLUMN + '.csv'
            # test_df.to_csv(csv_path)
            # print("Generations saved at path: ", path)

        torch.cuda.empty_cache()

# ------------------------------------------------------------ MAIN MODEL ------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='roberta', type=str, help='Choose from [roberta, FLAN-T5]')
    parser.add_argument('--file_path', default='./', type=str, help='The path containing train/val/test.csv')
    args = parser.parse_args()

    TEXT_INPUT_PATH = "/".join(args.file_path.split("/")[:-1])
    DATASET = args.file_path.split("/")[-2]
    
    if args.model_type == 'roberta':
        SOURCE_MAX_LEN = 512
        SOURCE_PREFIX = ''
        print("Using RoBERTa")
        TOKENIZER = RobertaTokenizerFast.from_pretrained('FacebookAI/roberta-base', truncation_side='left')
        print("RoBERTa Tokenizer loaded...\n")
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
    
    lab2idx, train_dataset = set_up_data_loader(text_path = TEXT_INPUT_PATH + '/train.csv', set_type = 'train')
    if "val.csv" in os.listdir(args.file_path):
        _, val_dataset = set_up_data_loader(text_path = TEXT_INPUT_PATH + '/val.csv', set_type = 'val')
    else:
        _, val_dataset = set_up_data_loader(text_path = TEXT_INPUT_PATH + '/test.csv', set_type = 'val')
    _, test_dataset = set_up_data_loader(text_path = TEXT_INPUT_PATH + '/test.csv', set_type = 'test')
    
    # ------------------------------ MODEL SETUP ------------------------------ #
        
    if args.model_type == 'roberta':
        MODEL = RoBERTaModel('FacebookAI/roberta-base', num_classes=len(lab2idx.keys()))
        print("RoBERTa Model loaded...\n")
        print(MODEL)
        OUTPUT_DIR = "./Models/RoBERTa/"
        LEARNING_RATE = 5e-6

    elif args.model_type == 'flan-t5':
        MODEL = T5Model('google/flan-t5-base', num_classes=len(lab2idx.keys()))
        print("Flan-T5 Model loaded...\n")
        OUTPUT_DIR = "./Models/Flan-T5/"
        LEARNING_RATE = 3e-5
    
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
          train_data_loader=train_dataset,
          val_data_loader=val_dataset,
          test_data_loader=test_dataset,
          learning_rate=LEARNING_RATE,
          model_type=args.model_type,
          **gen_kwargs)
    
    print("Model Trained!")