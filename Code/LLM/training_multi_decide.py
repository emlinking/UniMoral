import os
import argparse
import numpy as np
import pandas as pd
import warnings
import random
import re
from tqdm import tqdm
import gc

from collections import Counter

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler
from transformers.modeling_utils import PreTrainedModel, unwrap_model

from transformers import (
    AutoTokenizer,
    LlamaModel,
    AdamW
)

from huggingface_hub import login
access_token = "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"
login(token = access_token)

# -------------------------------------------------------------- CONFIG -------------------------------------------------------------- #

def set_random_seed(seed: int):
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

# ------------------------------------------------------------- DATA UTILS -------------------------------------------------------------- #

def read_csv_data(path):
    data = pd.read_csv(path)
    if 'labels' in data.columns:
        data.rename(columns = {'labels':'label'}, inplace = True)
    del path
    gc.collect()
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
    del label
    gc.collect()
    return lab

def preprocess_dataset(text_path: str):
    dataset = read_csv_data(text_path)
    DATASET = text_path.split("/")[-2]
    NUM_INP = 1

    if DATASET in input_label_sets:
        source = [SOURCE_PREFIX + str(s)for s in dataset['input'].tolist()]
        model_inputs = TOKENIZER(source, max_length=SOURCE_MAX_LEN, padding='max_length', truncation=True)
        
        all_labels = dataset['label'].tolist()
        idx2lab = list(set(all_labels))
        lab2idx = {k:v for v,k in enumerate(idx2lab)}

        # wts = dict(Counter(all_labels))
        # weights = torch.zeros((len(idx2lab)), dtype=torch.long).to(DEVICE)
        # for k,v in wts.items():
        #     weights[lab2idx[k]] = max(wts.values())/v
        # print("weights: ", weights)

        target = [lab2idx[t] for t in dataset['label'].tolist()]

        model_inputs['input_ids'] = torch.tensor([i for i in model_inputs['input_ids']], dtype=torch.long, device=DEVICE)
        model_inputs['attention_mask'] = torch.tensor([a for a in model_inputs['attention_mask']], dtype=torch.long, device=DEVICE)
        model_inputs['labels'] = torch.tensor(target, dtype=torch.long, device=DEVICE)
        print(model_inputs['labels'].size())

    elif DATASET in input_labels_sets:
        source = [SOURCE_PREFIX + s for s in dataset['input'].tolist()]
        model_inputs = TOKENIZER(source, max_length=SOURCE_MAX_LEN, padding='max_length', truncation=True)

        all_labels = [get_multilabel_ready(x) for x in dataset['label']]
        idx2lab = list(set([x for xs in all_labels for x in xs]))
        lab2idx = {k:v for v,k in enumerate(idx2lab)}

        # wts = dict(Counter(all_labels))
        # weights = torch.zeros((len(idx2lab)), dtype=torch.long).to(DEVICE)
        # for k,v in wts.items():
        #     weights[lab2idx[k]] = max(wts.values())/v
        # print("weights: ", weights)

        target = MultiLabelBinarizer().fit_transform(all_labels)

        model_inputs['input_ids'] = torch.tensor([i for i in model_inputs['input_ids']], dtype=torch.long, device=DEVICE)
        model_inputs['attention_mask'] = torch.tensor([a for a in model_inputs['attention_mask']], dtype=torch.long, device=DEVICE)
        model_inputs['labels'] = torch.tensor(target, dtype=torch.long, device=DEVICE)
        print(model_inputs['labels'].size())

    elif DATASET in inputs_label_sets:
        all_labels = dataset['label'].tolist()
        idx2lab = list(set(all_labels))
        lab2idx = {k:v for v,k in enumerate(idx2lab)}

        # wts = dict(Counter(all_labels))
        # weights = torch.zeros((len(idx2lab)), dtype=torch.float).to(DEVICE)
        # for k,v in wts.items():
        #     weights[lab2idx[k]] = float(max(wts.values())/v)
        # print("weights: ", weights)

        target = [lab2idx[t] for t in dataset['label'].tolist()]

        sources_input_ids = []
        sources_attn_mask = []
        inp_cols = [x for x in dataset.columns if x != 'label' and "Unnamed" not in x]
        NUM_INP = len(inp_cols)
        for inp in inp_cols:
            source = [SOURCE_PREFIX + s for s in dataset[inp].tolist()]
            source = TOKENIZER(source, max_length=SOURCE_MAX_LEN, padding='max_length', truncation=True)
            source['input_ids'] = torch.tensor([i for i in source['input_ids']], dtype=torch.long, device=DEVICE)
            source['attention_mask'] = torch.tensor([a for a in source['attention_mask']], dtype=torch.long, device=DEVICE)
            sources_input_ids.append(source['input_ids'])
            sources_attn_mask.append(source['attention_mask'])
        
        model_inputs = {
            'input_ids': torch.stack(sources_input_ids, dim=1),     ## seq x num_inp x source_max_len
            'attention_mask': torch.stack(sources_attn_mask,dim=1),
            'labels': torch.tensor(target, dtype=torch.long, device=DEVICE)
        }

        print(model_inputs['labels'].size())

    del text_path 
    del dataset
    del source
    del all_labels
    del target
    gc.collect()

    return model_inputs, lab2idx, NUM_INP

def set_up_data_loader(text_path1: str, text_path2: str, text_path3: str, set_type: str):
    dataset1, lab2idx1, NUM_INP1 = preprocess_dataset(text_path=text_path1)
    dataset2, lab2idx2, NUM_INP2 = preprocess_dataset(text_path=text_path2)
    dataset3, lab2idx3, NUM_INP3 = preprocess_dataset(text_path=text_path3)

    if set_type == 'train':
        dataset1, dataset2, dataset3 = process_dataset(dataset1, dataset2, dataset3)

    dataset1 = TensorDataset(dataset1['input_ids'], dataset1['attention_mask'], dataset1['labels'])
    dataset2 = TensorDataset(dataset2['input_ids'], dataset2['attention_mask'], dataset2['labels'])
    dataset3 = TensorDataset(dataset3['input_ids'], dataset3['attention_mask'], dataset3['labels'])

    if set_type == 'test':         ## No shuffling for test set
        return lab2idx1, lab2idx2, lab2idx3, NUM_INP1, NUM_INP2, NUM_INP3, DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=False), DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=False), DataLoader(dataset3, batch_size=BATCH_SIZE, shuffle=False)
    return lab2idx1, lab2idx2, lab2idx3, NUM_INP1, NUM_INP2, NUM_INP3, DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=True), DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=True), DataLoader(dataset3, batch_size=BATCH_SIZE, shuffle=True)

def process_dataset(dataset1, dataset2, dataset3):
    max_len = max(len(dataset1['input_ids']), len(dataset2['input_ids']), len(dataset3['input_ids']))

    if len(dataset1['input_ids']) < max_len: 
        diff = max_len - len(dataset1['input_ids'])
        new_idx = np.random.choice(list(range(len(dataset1['input_ids']))), size=diff, replace=True)
        new_inp_id = []
        new_attn_mask = []
        new_target = []
        for i in new_idx:
            new_inp_id.append(dataset2['input_ids'][i].unsqueeze(0))
            new_attn_mask.append(dataset2['attention_mask'][i].unsqueeze(0))
            new_target.append(dataset2['labels'][i].unsqueeze(0))
        new_inp_id = torch.cat(new_inp_id, 0)
        new_attn_mask = torch.cat(new_attn_mask, 0)
        new_target = torch.cat(new_target, 0)

        dataset1['input_ids'] = torch.cat((dataset1['input_ids'], new_inp_id), 0)
        dataset1['attention_mask'] = torch.cat((dataset1['attention_mask'], new_attn_mask), 0)
        dataset1['labels'] = torch.cat((dataset1['labels'], new_target))

    if len(dataset2['input_ids']) < max_len: 
        diff = max_len - len(dataset2['input_ids'])
        new_idx = np.random.choice(list(range(len(dataset2['input_ids']))), size=diff, replace=True)
        new_inp_id = []
        new_attn_mask = []
        new_target = []
        for i in new_idx:
            new_inp_id.append(dataset2['input_ids'][i].unsqueeze(0))
            new_attn_mask.append(dataset2['attention_mask'][i].unsqueeze(0))
            new_target.append(dataset2['labels'][i].unsqueeze(0))
        new_inp_id = torch.cat(new_inp_id, 0)
        new_attn_mask = torch.cat(new_attn_mask, 0)
        new_target = torch.cat(new_target, 0)

        dataset2['input_ids'] = torch.cat((dataset2['input_ids'], new_inp_id), 0)
        dataset2['attention_mask'] = torch.cat((dataset2['attention_mask'], new_attn_mask), 0)
        dataset2['labels'] = torch.cat((dataset2['labels'], new_target), 0)

    if len(dataset3['input_ids']) < max_len: 
        diff = max_len - len(dataset3['input_ids'])
        new_idx = np.random.choice(list(range(len(dataset3['input_ids']))), size=diff, replace=True)
        new_inp_id = []
        new_attn_mask = []
        new_target = []
        for i in new_idx:
            new_inp_id.append(dataset3['input_ids'][i].unsqueeze(0))
            new_attn_mask.append(dataset3['attention_mask'][i].unsqueeze(0))
            new_target.append(dataset3['labels'][i].unsqueeze(0))
        new_inp_id = torch.cat(new_inp_id, 0)
        new_attn_mask = torch.cat(new_attn_mask, 0)
        new_target = torch.cat(new_target, 0)

        dataset3['input_ids'] = torch.cat((dataset3['input_ids'], new_inp_id), 0)
        dataset3['attention_mask'] = torch.cat((dataset3['attention_mask'], new_attn_mask), 0)
        dataset3['labels'] = torch.cat((dataset3['labels'], new_target), 0)

    return dataset1, dataset2, dataset3


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

    del reference_list
    del hypothesis_list
    gc.collect()
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
                torch.save(state_dict, os.path.join(output_dir, DATASET1 + "_" + DATASET2 + "_" + DATASET3))
        else:
            model.save_pretrained(output_dir, state_dict=state_dict)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

def save_model(model, output_dir: str, tokenizer=None, state_dict=None):
        _save(model,output_dir, tokenizer=tokenizer, state_dict=state_dict)

def load_model(model, input_dir: str):
    state_dict = torch.load(input_dir, map_location = DEVICE)
    model.load_state_dict(state_dict)
    return model

# ----------------------------------------------------- MODEL ----------------------------------------------------- #

class MyLlamaModelMultiTaskDecide(nn.Module):
    def __init__(self, pretrained_model, num_inp1, num_inp2, num_inp3, num_classes_T1 = 2, num_classes_T2 = 2, num_classes_T3 = 2):
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
            nn.Dropout(p=0.1, inplace=False)
        )
        self.task1_classify = nn.Linear(num_inp1*768, num_classes_T1)

        self.task2_head = nn.Sequential(
            nn.LayerNorm(4096),
            nn.Linear(4096, 768), 
            nn.LayerNorm(768),
            nn.Dropout(p=0.1, inplace=False)
        )
        self.task2_classify = nn.Linear(num_inp2*768, num_classes_T2)

        self.task3_head = nn.Sequential(
            nn.LayerNorm(4096),
            nn.Linear(4096, 768), 
            nn.LayerNorm(768),
            nn.Dropout(p=0.1, inplace=False)
        )
        self.task3_classify = nn.Linear(num_inp3*768, num_classes_T3)
        
    def forward(self, source1=[], source2=[], source3=[]):
        op1, op2, op3 = [], [], []
        # print("source1['input_ids']: ", source1['input_ids'].size())
        # print("source1['attention_mask']: ", source1['attention_mask'].size())

        # print("source2['input_ids']: ", source2['input_ids'].size())
        # print("source2['attention_mask']: ", source2['attention_mask'].size())

        # print("source3['input_ids']: ", source3['input_ids'].size())
        # print("source3['attention_mask']: ", source3['attention_mask'].size())
        
        if source1:
            if DATASET1 in input_label_sets:
                op = self.encoder(source1['input_ids'], source1['attention_mask']).last_hidden_state.mean(dim=1)
                op = self.task1_head(op)
                op1 = self.task1_classify(op)

            elif DATASET1 in inputs_label_sets:
                op = []
                source1['input_ids'] = torch.transpose(source1['input_ids'], 0, 1)
                source1['attention_mask'] = torch.transpose(source1['attention_mask'], 0, 1)
                for input_ids, attention_mask in zip(source1['input_ids'], source1['attention_mask']):
                    op.append(self.task1_head(self.encoder(input_ids, attention_mask).last_hidden_state.mean(dim=1)))
                op = torch.cat(op, dim = 1)
                op1 = self.task1_classify(op)

            elif DATASET1 in input_labels_sets:
                op = self.encoder(source1['input_ids'], source1['attention_mask']).last_hidden_state.mean(dim=1)
                op = self.task1_head(op)
                op1 = self.task1_classify(op)
            
            for item in source1.items():
                del item

        if source2:
            if DATASET2 in input_label_sets:
                op = self.encoder(source2['input_ids'], source2['attention_mask']).last_hidden_state.mean(dim=1)
                op = self.task2_head(op)
                op2 = self.task2_classify(op)

            elif DATASET2 in inputs_label_sets:
                op = []
                source2['input_ids'] = torch.transpose(source2['input_ids'], 0, 1)
                source2['attention_mask'] = torch.transpose(source2['attention_mask'], 0, 1)
                for input_ids, attention_mask in zip(source2['input_ids'], source2['attention_mask']):
                    op.append(self.task2_head(self.encoder(input_ids, attention_mask).last_hidden_state.mean(dim=1)))
                op = torch.cat(op, dim = 1)
                op2 = self.task2_classify(op)

            elif DATASET2 in input_labels_sets:
                op = self.encoder(source2['input_ids'], source2['attention_mask']).last_hidden_state.mean(dim=1)
                op = self.task2_head(op)
                op2 = self.task2_classify(op)

            for item in source2.items():
                del item

        if source3:
            if DATASET3 in input_label_sets:
                op = self.encoder(source3['input_ids'], source3['attention_mask']).last_hidden_state.mean(dim=1)
                op = self.task3_head(op)
                op3 = self.task3_classify(op)

            elif DATASET3 in inputs_label_sets:
                op = []
                source3['input_ids'] = torch.transpose(source3['input_ids'], 0, 1)
                source3['attention_mask'] = torch.transpose(source3['attention_mask'], 0, 1)
                for input_ids, attention_mask in zip(source3['input_ids'], source3['attention_mask']):
                    op.append(self.task3_head(self.encoder(input_ids, attention_mask).last_hidden_state.mean(dim=1)))
                op = torch.cat(op, dim = 1)
                op3 = self.task3_classify(op)

            elif DATASET3 in input_labels_sets:
                op = self.encoder(source3['input_ids'], source3['attention_mask']).last_hidden_state.mean(dim=1)
                op = self.task3_head(op)
                op3 = self.task3_classify(op)

            for item in source3.items():
                del item

        del source1
        del source2
        del source3
        gc.collect()

        if type(op1) == torch.Tensor and type(op2) == torch.Tensor and type(op3) == torch.Tensor:
            return op1, op2, op3
        elif type(op1) == torch.Tensor:
            return op1
        elif type(op2) == torch.Tensor:
            return op2
        elif type(op3) == torch.Tensor:
            return op3
            
# ----------------------------------------------------- TRAINING UTILS ----------------------------------------------------- #

def prepare_for_training(model, learning_rate):
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterion3 = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    return optimizer, criterion1, criterion2, criterion3

def train_epoch(model, data_loader1, data_loader2, data_loader3, optimizer, criterion1, criterion2, criterion3):
    model.train()
    epoch_train_loss = 0.0
    for step, (batch1, batch2, batch3) in enumerate(tqdm(zip(data_loader1, data_loader2, data_loader3), desc="Training Iteration")):
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

        input_ids, attention_mask, target3 = batch3
        source3 = {
            'input_ids': input_ids.to(DEVICE),
            'attention_mask': attention_mask.to(DEVICE)
        }

        target1 = target1.to(DEVICE)
        target2 = target2.to(DEVICE)
        target3 = target3.to(DEVICE)
        optimizer.zero_grad()

        logits1, logits2, logits3 = model(source1, source2, source3)
        
        loss1 = criterion1(logits1, target1)
        loss2 = criterion2(logits2, target2)
        loss3 = criterion3(logits3, target3)

        loss = (loss1+loss2+loss3)/3
        epoch_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        del loss1
        del loss2
        del loss3
        del loss
        del input_ids
        del attention_mask
        del logits1
        del logits2
        del logits3
        del target1
        del target2
        del target3
        for item in source1.items():
            del item
        for item in source2.items():
            del item
        for item in source3.items():
            del item
        del source1
        del source2
        del source3
        gc.collect()
        torch.cuda.empty_cache()

        # if step == 2:
        #     break

    del batch1
    del batch2
    del batch3
    gc.collect()
    torch.cuda.empty_cache()
    
    return epoch_train_loss/ step

def val_epoch(model, data_loader1, data_loader2, data_loader3, criterion1, criterion2, criterion3):
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for step, (batch1, batch2, batch3) in enumerate(tqdm(zip(data_loader1, data_loader2, data_loader3), desc="Validation Iteration")):
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

            input_ids, attention_mask, target3 = batch3
            source3 = {
                'input_ids': input_ids.to(DEVICE),
                'attention_mask': attention_mask.to(DEVICE)
            }

            target1 = target1.to(DEVICE)
            target2 = target2.to(DEVICE)
            target3 = target3.to(DEVICE)

            logits1, logits2, logits3 = model(source1, source2, source3)

            loss1 = criterion1(logits1, target1)
            loss2 = criterion2(logits2, target2)
            loss3 = criterion3(logits3, target3)

            loss = (loss1+loss2+loss3)/3
            epoch_val_loss += loss.item()  

            del loss1
            del loss2
            del loss3
            del loss
            del input_ids
            del attention_mask
            del logits1
            del logits2
            del logits3
            del target1
            del target2
            del target3
            for item in source1.items():
                del item
            for item in source2.items():
                del item
            for item in source3.items():
                del item
            del source1
            del source2
            del source3
            gc.collect()
            torch.cuda.empty_cache()
            # if step == 2:
            #     break

    del batch1
    del batch2
    del batch3
    del batch4
    gc.collect()
    torch.cuda.empty_cache() 
    
    return epoch_val_loss/ step

def test_epoch(model, data_loader1, data_loader2, data_loader3, desc):
    model.eval()
    predictions1, predictions2, predictions3 = [], [], []
    gold1, gold2, gold3 = [], [], []
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

            del batch1
            del data_loader1
            del input_ids
            del attention_mask
            del target1
            del logits1
            for item in source1.items():
                del item
            del source1
            del predicted_classes1

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

            del batch2
            del data_loader2
            del input_ids
            del attention_mask
            del target2
            del logits2
            for item in source2.items():
                del item
            del source2
            del predicted_classes2

        for step, batch3 in enumerate(tqdm(data_loader3, desc=desc)):
            input_ids, attention_mask, target3 = batch3
            source3 = {
                'input_ids': input_ids.to(DEVICE),
                'attention_mask': attention_mask.to(DEVICE)
            }

            target3 = target3.to(DEVICE)
            logits3 = model(source3 = source3)

            predicted_classes3 = torch.argmax(nn.Softmax(dim=1)(logits3), dim=1)
            predictions3.extend(predicted_classes3.tolist())
            gold3.extend(target3.tolist())

            del batch3
            del data_loader3
            del input_ids
            del attention_mask
            del target3
            del logits3
            for item in source3.items():
                del item
            del source3
            del predicted_classes3

        gc.collect()
        torch.cuda.empty_cache()
        # if step == 2:
        #     break

    gc.collect()
    torch.cuda.empty_cache()
    return predictions1, gold1, predictions2, gold2, predictions3, gold3

def compute_metrics(model, data_loader1, data_loader2, data_loader3, desc):

    predictions1, gold1, predictions2, gold2, predictions3, gold3 = test_epoch(model, data_loader1, data_loader2, data_loader3, desc=desc)
    result1 = get_scores(gold1, predictions1)
    result2 = get_scores(gold2, predictions2)
    result3 = get_scores(gold3, predictions3)

    torch.cuda.empty_cache() 
    
    return predictions1, gold1, predictions2, gold2, predictions3, gold3, result1, result2, result3

def train(model, tokenizer, train_data_loader1, train_data_loader2, train_data_loader3, val_data_loader1, val_data_loader2, val_data_loader3, test_data_loader1, test_data_loader2, test_data_loader3, learning_rate, model_type):
    train_losses = []
    val_losses = []
    
    optimizer, criterion1, criterion2, criterion3 = prepare_for_training(model=model, learning_rate=learning_rate)
    
    min_val_loss = 99999
    bad_loss = 0
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_data_loader1, train_data_loader2, train_data_loader3, optimizer, criterion1, criterion2, criterion3)
        train_losses.append(train_loss)
        print("Epoch: {}\ttrain_loss: {}".format(epoch+1, train_loss))

        val_loss = val_epoch(model, val_data_loader1, val_data_loader2, val_data_loader3, criterion1, criterion2, criterion3)
        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            bad_loss = 0
            min_val_loss = val_loss
            test_pred1, test_gold1, test_pred2, test_gold2, test_pred3, test_gold3, test_results1, test_results2, test_results3 = compute_metrics(model, tokenizer, test_data_loader1, test_data_loader2, test_data_loader3, desc="Test Iteration")
            
            print("Epoch: {}\ttrain_loss: {}\tval_loss: {}\tmin_val_loss: {}".format(epoch+1, train_loss, val_loss, min(val_losses)))
            print("\nTest Accuracy for DATASET ", DATASET1, ": ", test_results1['accuracy'])
            print("\nTest Classification Report for DATASET ", DATASET1, ":\n", test_results1['report'])

            print("\nTest Accuracy for DATASET ", DATASET2, ": ", test_results2['accuracy'])
            print("\nTest Classification Report for DATASET ", DATASET2, ":\n", test_results2['report'])

            print("\nTest Accuracy for DATASET ", DATASET3, ": ", test_results3['accuracy'])
            print("\nTest Classification Report for DATASET ", DATASET3, ":\n", test_results3['report'])
        
            path = OUTPUT_DIR + DATASET1 + "_" + DATASET2 + "_" + DATASET3 + "_" + LOAD_MODEL
            save_model(model, path, tokenizer)
            print("Model saved at path: ", path)

            # test_df = pd.DataFrame()
            # test_df['Predictions'] = test_pred
            # test_df['Gold'] = test_gold

            # csv_path = OUTPUT_DIR + model_type + "_"  + DATASET + "_" + LOAD_MODEL + '.csv'
            # test_df.to_csv(csv_path)
            # print("Predictions saved at path: ", csv_path)
        else:
            bad_loss += 1
        
        if bad_loss == EARLY_STOPPING_THRESHOLD:
            print("Stopping early...")
            break

        torch.cuda.empty_cache()
    torch.cuda.empty_cache()

# def infer(model, tokenizer, test_data_loader, model_type, save_csv=False):
#     test_pred, test_gold, test_results = compute_metrics(model, tokenizer, test_data_loader, desc="Test Iteration")
            
#     print("\nTest Accuracy: ", test_results['accuracy'])
#     print("\nTest Classification Report:\n", test_results['report'])

#     if save_csv:
#         test_df = pd.DataFrame()
#         test_df['Predictions'] = test_pred
#         test_df['Gold'] = test_gold

#         csv_path = OUTPUT_DIR + model_type + "_"  + DATASET + "_" + LOAD_MODEL + '.csv'
#         test_df.to_csv(csv_path)
#         print("Predictions saved at path: ", csv_path)

#     torch.cuda.empty_cache()

# ------------------------------------------------------------ MAIN MODEL ------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='llama_3', type=str, help='Choose from [llama-3]')
    parser.add_argument('--file_pathT1', default='/shared/0/Morality/OnlyFiles/dilemmas/', type=str, help='The path containing train/val/test.csv for data 1')
    parser.add_argument('--file_pathT2', default='/shared/0/Morality/OnlyFiles/ethics/utilitarianism/', type=str, help='The path containing train/val/test.csv for data 2')
    parser.add_argument('--file_pathT3', default='/shared/0/Morality/OnlyFiles/moralstories/action_classification/', type=str, help='The path containing train/val/test.csv for data 3')
    parser.add_argument('--batch_size', default=4, type=int, help='The batch size for training')
    parser.add_argument('--load_model', default='None', type=str, help='The path containing the pretrained model to load. None if no model.')
    # parser.add_argument('--only_eval', default='FALSE', type=str, help='Set to "TRUE" if only have to perform evaluation on test set.')
    parser.add_argument('--device', default='cuda:0', type=str, help='Tell which device to run the code on?')
    args = parser.parse_args()

    DEVICE = torch.device(args.device)

    SEED = 42
    set_random_seed(SEED)

    TARGET_MAX_LEN = 1

    MAX_EPOCHS = 10

    EARLY_STOPPING_THRESHOLD = 2

    SCALER = GradScaler()

    input_label_sets = ['anecdotes', 'action_classification', 'consequence_classification', 'commonsense', 'deontology', 'justice', 'virtue', 'moralexceptqa', 'moralfoundationredditcorpus', 'storycommonsense', 'moralconvita', 'moralfoundationstwittercorpus']
    input_labels_sets = ['moralintegritycorpus']
    inputs_label_sets = ['dilemmas', 'utilitarianism', 'storal_en', 'storal_zh']

    TEXT_INPUT_PATH_T1 = "/".join(args.file_pathT1.split("/")[:-1])
    TEXT_INPUT_PATH_T2 = "/".join(args.file_pathT2.split("/")[:-1])
    TEXT_INPUT_PATH_T3 = "/".join(args.file_pathT3.split("/")[:-1])
    DATASET1 = args.file_pathT1.split("/")[-2]
    DATASET2 = args.file_pathT2.split("/")[-2]
    DATASET3 = args.file_pathT3.split("/")[-2]

    LOAD_MODEL = ""
    BATCH_SIZE = args.batch_size
    
    if args.model_type == 'llama_3':
        SOURCE_MAX_LEN = 1024
        SOURCE_PREFIX = ''
        print("Using Llama 3")
        if args.load_model != "None":
            TOKENIZER = AutoTokenizer.from_pretrained("/".join(args.load_model.split("/")[:-1]), truncation_side='left')
            LOAD_MODEL = args.load_model.split("/")[-1]
        else:
            TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", truncation_side='left')
        TOKENIZER.pad_token = TOKENIZER.eos_token
        print("Llama 3 Tokenizer loaded...\n")
    else:
        print("Error: Wrong model type")
        exit(0)
    # ------------------------------ READ DATASET ------------------------------ #
    
    lab2idx1, lab2idx2, lab2idx3, NUM_INP1, NUM_INP2, NUM_INP3, train_dataset1, train_dataset2, train_dataset3 = set_up_data_loader(
        text_path1 = TEXT_INPUT_PATH_T1 + '/train.csv',
        text_path2 = TEXT_INPUT_PATH_T2 + '/train.csv',
        text_path3 = TEXT_INPUT_PATH_T3 + '/train.csv',
        set_type = 'train'
    )

    _1, _2, _3, __1, __2, __3, val_dataset1, val_dataset2, val_dataset3 = set_up_data_loader(
        text_path1 = TEXT_INPUT_PATH_T1 + '/val.csv',
        text_path2 = TEXT_INPUT_PATH_T2 + '/test.csv',
        text_path3 = TEXT_INPUT_PATH_T3 + '/val.csv',
        set_type = 'val'
    )

    _1, _2, _3, __1, __2, __3, test_dataset1, test_dataset2, test_dataset3 = set_up_data_loader(
        text_path1 = TEXT_INPUT_PATH_T1 + '/test.csv',
        text_path2 = TEXT_INPUT_PATH_T2 + '/test.csv',
        text_path3 = TEXT_INPUT_PATH_T3 + '/test.csv',
        set_type = 'test'
    )

    print(f"Train datasets: {len(train_dataset1)}, {len(train_dataset2)}, {len(train_dataset3)}")
    print(f"Val datasets: {len(val_dataset1)}, {len(val_dataset2)}, {len(val_dataset3)}")
    print(f"Test datasets: {len(test_dataset1)}, {len(test_dataset2)}, {len(test_dataset3)}")
    # ------------------------------ MODEL SETUP ------------------------------ #

    if args.model_type == 'llama_3':
        MODEL = MyLlamaModelMultiTaskDecide(
            pretrained_model = "meta-llama/Meta-Llama-3-8B",
            num_inp1 = NUM_INP1,
            num_inp2 = NUM_INP2,
            num_inp3 = NUM_INP3,
            num_classes_T1 = len(lab2idx1.keys()),
            num_classes_T2 = len(lab2idx2.keys()),
            num_classes_T3 = len(lab2idx3.keys())
        )
        print("Llama Model loaded...\n")
        print(MODEL)
        OUTPUT_DIR = "./models/llama_3/multitask/"
        if args.load_model != "None":
            print("Loading model ", args.load_model, "...")
            MODEL = load_model(MODEL, args.load_model)
            print("Model State Dict Loaded")
        LEARNING_RATE = 5e-6
    
    else:
        print("Error: Wrong model type")
        exit(0)

    MODEL.to(DEVICE)
    print(LEARNING_RATE)
    print(OUTPUT_DIR)
    print(SOURCE_PREFIX)
    
    # ------------------------------ TRAINING SETUP ------------------------------ #
    # if args.only_eval == "TRUE":
    #     infer(model=MODEL,
    #         tokenizer=TOKENIZER,
    #         test_data_loader=test_dataset,
    #         model_type=args.model_type,
    #         save_csv=True
    #     )
    # else:
    train(model=MODEL,
        tokenizer=TOKENIZER,
        train_data_loader1=train_dataset1,
        train_data_loader2=train_dataset2,
        train_data_loader3=train_dataset3,
        val_data_loader1=val_dataset1,
        val_data_loader2=val_dataset2,
        val_data_loader3=val_dataset3,
        test_data_loader1=test_dataset1,
        test_data_loader2=test_dataset2,
        test_data_loader3=test_dataset3,
        learning_rate=LEARNING_RATE,
        model_type=args.model_type
    )
    
    print("Model Trained!")