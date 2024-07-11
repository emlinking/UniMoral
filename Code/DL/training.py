import warnings
warnings.filterwarnings('ignore')

import sys
import re
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, FastText
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = "cuda:0"

train_path = sys.argv[1]
val_path = sys.argv[2]

try:
    test_path = sys.argv[3]
except:
    test_path = sys.argv[2]     ## Validation is test set

train_data = pd.read_csv(train_path)
valid_data = pd.read_csv(val_path)
test_data = pd.read_csv(test_path)

if 'labels' in train_data.columns:
    train_data.rename(columns = {'labels':'label'}, inplace = True)
if 'labels' in valid_data.columns:
    valid_data.rename(columns = {'labels':'label'}, inplace = True)
if 'labels' in test_data.columns:
    test_data.rename(columns = {'labels':'label'}, inplace = True)

input_label_sets = ['anecdotes', 'action_classification', 'consequence_classification', 'commonsense', 'deontology', 'justice', 'virtue', 'moralexceptqa', 'moralfoundationredditcorpus', 'storycommonsense', 'moralconvita', 'moralfoundationstwittercorpus']
input_labels_sets = ['moralintegritycorpus']
inputs_label_sets = ['dilemmas', 'utilitarianism', 'storal_en', 'storal_zh']

def tokenize(text):
    text = str(text)
    tokens = text.split()
    return tokens

# tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
tokenizer = tokenize

# Build Vocabulary
def yield_tokens(data):
    for col in data.columns:
        for text in data[col]:
            text = str(text)
            yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

# Load FastText embeddings
# fasttext_vectors = FastText(language='en')
# vocab.vocab.vectors(fasttext_vectors)

PAD_IDX = vocab["<pad>"]
UNK_IDX = vocab["<unk>"]

def get_multilabel_ready(label):
    lab = str(label)
    lab = re.sub("\[","", label)
    lab = re.sub("\]","", lab)
    lab = re.sub("\"","", lab)
    lab = re.sub("\'","", lab)
    lab = [x.strip() for x in lab.split(",")]
    return lab

class TextDataset_One_Input_One_Output(Dataset):
    def __init__(self, data, vocab, tokenizer):
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer

        self.idx2lab = list(set(data['label'].tolist()))
        self.lab2idx = {k:v for v,k in enumerate(self.idx2lab)}

        print("Labels: ", self.lab2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['input']
        label = self.lab2idx[self.data.iloc[idx]['label']]
        tokenized_text = [self.vocab[token] for token in self.tokenizer(text)]
        return torch.tensor(tokenized_text), torch.tensor(label, dtype=torch.float)
    
class TextDataset_One_Input_Multi_Output(Dataset):
    def __init__(self, data, vocab, tokenizer):
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer

        all_labels = []
        for label in data['label']:
            label = get_multilabel_ready(label)
            all_labels.append(label)
        
        self.idx2lab = list(set([x for xs in all_labels for x in xs]))
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.idx2lab])
        self.lab2idx = {k:v for v,k in enumerate(self.mlb.classes_)}

        print("Labels: ", self.lab2idx)
        
    def get_mlb(self):
        return self.mlb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['input']
        label = get_multilabel_ready(self.data.iloc[idx]['label'])
        label = self.mlb.transform([label])
        tokenized_text = [self.vocab[token] for token in self.tokenizer(text)]
        return torch.tensor(tokenized_text), torch.tensor(label, dtype=torch.float)
    
class TextDataset_Multi_Input_One_Output(Dataset):
    def __init__(self, data, vocab, tokenizer):
        self.data = data
        self.input_cols = [x for x in self.data.columns if x != 'label' and "Unnamed" not in x]
        self.num_input_cols = len(self.input_cols)
        self.vocab = vocab
        self.tokenizer = tokenizer

        self.idx2lab = list(set(data['label'].tolist()))
        self.lab2idx = {k:v for v,k in enumerate(self.idx2lab)}

        print("Labels: ", self.lab2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.lab2idx[self.data.iloc[idx]['label']]
        columns = [self.data.iloc[idx][x] for x in self.input_cols]
        tokenized_texts = []
        for col in columns:
            tokens = [self.vocab[token] for token in self.tokenizer(col)]
            tokenized_texts.append(torch.tensor(tokens))

        return tokenized_texts, torch.tensor(label, dtype=torch.float)

def collate_batch(batch):
    if DATASET in inputs_label_sets:
        label_list = []
        label_list = [x[-1] for x in batch]

        num_inp = len(batch[0][0])
        text_list = [[]]*num_inp
        lengths = [[]]*num_inp

        for i in range(num_inp):
            texts = [x[0][i] for x in batch]
            text_list[i] = []
            lengths[i] = []
            for _text in texts:
                processed_text = _text.clone().detach()
                text_list[i].append(processed_text)
                lengths[i].append(processed_text.size(0))
            text_list[i] = pad_sequence(text_list[i], padding_value=PAD_IDX)
            lengths[i] = torch.tensor(lengths[i])
        return text_list, lengths, torch.tensor(label_list)
    
    label_list, text_list, lengths = [], [], []
    for (_text, _label) in batch:
        if DATASET in input_labels_sets:
            label_list.append(_label[0].tolist())
        else:
            label_list.append(_label)
        processed_text = _text.clone().detach()
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    text_list = pad_sequence(text_list, padding_value=PAD_IDX)

    return text_list, torch.tensor(lengths), torch.tensor(label_list)


## folder name where train/val/test.csv are there
DATASET = train_path.split("/")[-2]

# Create datasets
if DATASET in input_label_sets:
    train_dataset = TextDataset_One_Input_One_Output(train_data, vocab, tokenizer)
    valid_dataset = TextDataset_One_Input_One_Output(valid_data, vocab, tokenizer)
    test_dataset = TextDataset_One_Input_One_Output(test_data, vocab, tokenizer)

elif DATASET in input_labels_sets:
    train_dataset = TextDataset_One_Input_Multi_Output(train_data, vocab, tokenizer)
    valid_dataset = TextDataset_One_Input_Multi_Output(valid_data, vocab, tokenizer)
    test_dataset = TextDataset_One_Input_Multi_Output(test_data, vocab, tokenizer)
    multi_label_mlb = train_dataset.get_mlb()

elif DATASET in inputs_label_sets:
    train_dataset = TextDataset_Multi_Input_One_Output(train_data, vocab, tokenizer)
    valid_dataset = TextDataset_Multi_Input_One_Output(valid_data, vocab, tokenizer)
    test_dataset = TextDataset_Multi_Input_One_Output(test_data, vocab, tokenizer)
else:
    print("Wrong dataset!")

BATCH_SIZE = 16
INPUT_DIM = len(vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
OUTPUT_DIM = len(train_dataset.idx2lab)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
THRESHOLD_PROB = 0.09

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        if DATASET in inputs_label_sets:
            num_inp = len(text)
            outputs = []

            for i in range(num_inp):
                input = text[i]
                input = torch.stack(input)
                embedded = self.dropout(self.embedding(input))
                packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, torch.stack(text_lengths[i]).cpu(), enforce_sorted=False)
                packed_output, (hidden, cell) = self.lstm(packed_embedded)
                outputs.append(hidden)
            hidden = torch.stack(outputs).mean(dim=0)
        else:
            embedded = self.dropout(self.embedding(text))
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), enforce_sorted=False)
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        
        return self.fc(hidden)

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.gru = nn.GRU(embedding_dim, 
                          hidden_dim, 
                          num_layers=n_layers, 
                          bidirectional=bidirectional, 
                          dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        if DATASET in inputs_label_sets:
            num_inp = len(text)
            outputs = []

            for i in range(num_inp):
                input = text[i]
                input = torch.stack(input)
                embedded = self.dropout(self.embedding(input))
                packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, torch.stack(text_lengths[i]).cpu(), enforce_sorted=False)
                packed_output, hidden = self.gru(packed_embedded)
                outputs.append(hidden)
            hidden = torch.stack(outputs).mean(dim=0)
        else:
            embedded = self.dropout(self.embedding(text))
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), enforce_sorted=False)
            packed_output, hidden = self.gru(packed_embedded)
        
        if self.gru.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        
        return self.fc(hidden)

lstm_model = LSTMModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, 
                    BIDIRECTIONAL, DROPOUT, PAD_IDX)

gru_model = GRUModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, 
                    BIDIRECTIONAL, DROPOUT, PAD_IDX)

# Load pre-trained embeddings
# lstm_model.embedding.weight.data.copy_(vocab.vectors)
# gru_model.embedding.weight.data.copy_(vocab.vectors)

# Zero the initial weights of the unknown and padding tokens
lstm_model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
lstm_model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
gru_model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
gru_model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# Define the optimizer and loss function
optimizer_lstm = optim.Adam(lstm_model.parameters())
optimizer_gru = optim.Adam(gru_model.parameters())

if DATASET in input_labels_sets:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

lstm_model = lstm_model.to(device)
gru_model = gru_model.to(device)
criterion = criterion.to(device)

# Define training and evaluation functions
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in tqdm(iterator, desc="train loop"):
        optimizer.zero_grad()
        text, text_lengths, labels = batch
        if DATASET in input_label_sets:
            text, text_lengths, labels = text.to(device), text_lengths.to(device), labels.to(device)
            predictions = model(text, text_lengths).squeeze(1)
            labels = labels.type(torch.LongTensor)

            loss = criterion(predictions.to(device), labels.to(device))  
            pred_labels = torch.argmax(nn.Softmax(dim=1)(predictions), dim=1)
            acc = accuracy_score(pred_labels.cpu().numpy(), labels.cpu().numpy())

        elif DATASET in input_labels_sets:
            text, text_lengths, labels = text.to(device), text_lengths.to(device), labels.to(device)
            predictions = model(text, text_lengths).squeeze(1)
            labels = labels.type(torch.LongTensor)

            loss = criterion(predictions.float().to(device), labels.float().to(device))
            predictions = nn.Softmax(dim=1)(predictions)
            pred_labels = []
            for preds in predictions:
                prediction = []
                for pr in preds:
                    if pr > THRESHOLD_PROB:
                        prediction.append(1)
                    else:
                        prediction.append(0)
                pred_labels.append(prediction)

            acc = accuracy_score(pred_labels, labels)

        elif DATASET in inputs_label_sets:
            for i in range(len(text)):
                text[i] = [x.to(device) for x in text[i]]
                text_lengths[i] = [x.to(device) for x in text_lengths[i]]
            labels = labels.to(device)
            predictions = model(text, text_lengths).squeeze(1)
            labels = labels.type(torch.LongTensor)

            loss = criterion(predictions.to(device), labels.to(device))  
            pred_labels = torch.argmax(nn.Softmax(dim=1)(predictions), dim=1)
            acc = accuracy_score(pred_labels.cpu().numpy(), labels.cpu().numpy())

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    gold_labs = []
    model_labs = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluate loop"):
            text, text_lengths, labels = batch

            if DATASET in input_label_sets:
                text, text_lengths, labels = text.to(device), text_lengths.to(device), labels.to(device)
                predictions = model(text, text_lengths).squeeze(1)
                labels = labels.type(torch.LongTensor)

                loss = criterion(predictions.to(device), labels.to(device))  
                pred_labels = torch.argmax(nn.Softmax(dim=1)(predictions), dim=1)
                acc = accuracy_score(pred_labels.cpu().numpy(), labels.cpu().numpy())

                gold_labs.extend(labels.cpu().tolist())
                model_labs.extend(pred_labels.cpu().tolist())

            elif DATASET in input_labels_sets:
                text, text_lengths, labels = text.to(device), text_lengths.to(device), labels.to(device)
                predictions = model(text, text_lengths).squeeze(1)
                labels = labels.type(torch.LongTensor)

                loss = criterion(predictions.float().to(device), labels.float().to(device))
                predictions = nn.Softmax(dim=1)(predictions)
                pred_labels = []
                for preds in predictions:
                    prediction = []
                    for pr in preds:
                        if pr > THRESHOLD_PROB:
                            prediction.append(1)
                        else:
                            prediction.append(0)
                    pred_labels.append(prediction)

                acc = accuracy_score(pred_labels, labels)

                gold_labs.extend(labels.cpu().tolist())
                model_labs.extend(pred_labels.cpu().tolist())

            elif DATASET in inputs_label_sets:
                for i in range(len(text)):
                    text[i] = [x.to(device) for x in text[i]]
                    text_lengths[i] = [x.to(device) for x in text_lengths[i]]
                labels = labels.to(device)
                predictions = model(text, text_lengths).squeeze(1)
                labels = labels.type(torch.LongTensor)

                loss = criterion(predictions.to(device), labels.to(device))  
                pred_labels = torch.argmax(nn.Softmax(dim=1)(predictions), dim=1)
                acc = accuracy_score(pred_labels.cpu().numpy(), labels.cpu().numpy())

                gold_labs.extend(labels.cpu().tolist())
                model_labs.extend(pred_labels.cpu().tolist())

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            # print("model_labs: ", model_labs)
            # print("gold_labs: ", gold_labs)
    return epoch_loss / len(iterator), epoch_acc / len(iterator), classification_report(gold_labs,model_labs)

# Training loop
N_EPOCHS = 10
min_loss = 999

for epoch in tqdm(range(N_EPOCHS)):
    train_loss, train_acc = train(lstm_model, train_loader, optimizer_lstm, criterion)
    valid_loss, valid_acc, valid_report = evaluate(lstm_model, valid_loader, criterion)
    print(f'LSTM Epoch: {epoch+1}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%\n')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%\n')

    if valid_loss < min_loss:
        min_loss = valid_loss
        print("Minimum loss ", min_loss, " at epoch: ", epoch)
        test_loss_lstm, test_acc_lstm, test_report_lstm = evaluate(lstm_model, test_loader, criterion)
        print(f'LSTM Test Loss: {test_loss_lstm:.3f} | Test Acc: {test_acc_lstm*100:.2f}%\n')
        print("Classification report\n", test_report_lstm)

min_loss = 999
for epoch in tqdm(range(N_EPOCHS)):
    train_loss, train_acc = train(gru_model, train_loader, optimizer_gru, criterion)
    valid_loss, valid_acc, valid_report = evaluate(gru_model, valid_loader, criterion)
    print(f'GRU Epoch: {epoch+1}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%\n')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%\n')
    if valid_loss < min_loss:
        min_loss = valid_loss
        print("Minimum loss ", min_loss, " at epoch: ", epoch)
        test_loss_gru, test_acc_gru, test_report_gru = evaluate(gru_model, test_loader, criterion)
        print(f'GRU Test Loss: {test_loss_gru:.3f} | Test Acc: {test_acc_gru*100:.2f}%\n')
        print("Classification report\n", test_report_gru)