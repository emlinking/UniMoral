import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
import sys
import re

# from huggingface_hub import hf_hub_download
import fasttext

np.random.seed(42)

train_path = sys.argv[1]
test_path = sys.argv[2]

input_label_sets = ['anecdotes', 'action_classification', 'consequence_classification', 'commonsense', 'deontology', 'justice', 'virtue', 'moralexceptqa', 'moralfoundationredditcorpus', 'storycommonsense', 'moralconvita', 'moralfoundationstwittercorpus']
input_labels_sets = ['moralintegritycorpus']
inputs_label_sets = ['dilemmas', 'utilitarianism', 'storal_en', 'storal_zh']

def preprocess(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def load_fasttext_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

print("Loading Fasttext model...")
if "convita" in train_path:
    fasttext_embeddings = fasttext.load_model("/shared/0/Morality/italian_fasttext.bin")
elif "storal_zh" in train_path:
    fasttext_embeddings = fasttext.load_model("/shared/0/Morality/chinese_fasttext.bin")
else:
    fasttext_embeddings = load_fasttext_embeddings('/shared/0/resources/fasttext/crawl-300d-2M.vec')

def get_sentence_embedding(text, embeddings_index, embedding_dim=300):
    words = text.split()
    embeddings = [embeddings_index[word] for word in words if word in embeddings_index]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(embedding_dim)

def get_multilabel_ready(label):
    lab = str(label)
    lab = re.sub("\[","", label)
    lab = re.sub("\]","", lab)
    lab = re.sub("\"","", lab)
    lab = re.sub("\'","", lab)
    lab = [x.strip() for x in lab.split(",")]
    return lab

def get_data_ready(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    if 'labels' in train_data.columns:
        train_data.rename(columns = {'labels':'label'}, inplace = True)
    if 'labels' in test_data.columns:
        test_data.rename(columns = {'labels':'label'}, inplace = True)

    train_data['label'] = train_data['label'].astype(str)
    test_data['label'] = test_data['label'].astype(str)

    train_data = train_data[train_data.columns.drop(list(train_data.filter(regex='Unnamed')))]
    test_data = test_data[test_data.columns.drop(list(test_data.filter(regex='Unnamed')))]
    
    ## folder name where train/test.csv are there
    dataset = train_path.split("/")[-2]
    
    ## one input, one label
    if dataset in input_label_sets:
        X_train, y_train = train_data['input'].tolist(), train_data['label'].tolist()
        X_test, y_test = test_data['input'].tolist(), test_data['label'].tolist()
        X_train = [preprocess(text) for text in X_train]
        X_test = [preprocess(text) for text in X_test]
        X_train = [get_sentence_embedding(text, fasttext_embeddings) for text in tqdm(X_train, desc="Getting train embeds")]
        X_test = [get_sentence_embedding(text, fasttext_embeddings) for text in tqdm(X_test, desc="Getting test embeds")]

        # Convert labels to numerical format
        unique_labels = list(set(y_train).union(set(y_test)))
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        y_train = [label_to_index[label] for label in y_train]
        y_test = [label_to_index[label] for label in y_test]

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        print("Label to index: ", label_to_index)
        print("Class weights: ", class_weight_dict)

        svm_classifier = SVC(class_weight=class_weight_dict, max_iter=10000)
        lr_classifier = LogisticRegression(class_weight=class_weight_dict, max_iter=10000)

        return X_train, y_train, X_test, y_test, svm_classifier, lr_classifier

    ## one input, many labels
    elif dataset in input_labels_sets:
        X_train, X_test  = train_data['input'].tolist(), test_data['input'].tolist()
        X_train = [preprocess(text) for text in X_train]
        X_test = [preprocess(text) for text in X_test]
        X_train = [get_sentence_embedding(text, fasttext_embeddings) for text in tqdm(X_train, desc="Getting train embeds")]
        X_test = [get_sentence_embedding(text, fasttext_embeddings) for text in tqdm(X_test, desc="Getting test embeds")]
        
        y_train_tmp = [get_multilabel_ready(x) for x in train_data['label'].tolist()]
        y_test_tmp = [get_multilabel_ready(x) for x in test_data['label'].tolist()]
        
        # Convert labels to numerical format
        all_label_list = [x for lab in y_train_tmp for x in lab]
        all_label_list.extend([x for lab in y_test_tmp for x in lab])
        unique_labels = list(set(all_label_list))
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        y_train, y_test = [], []
        for lab in y_train_tmp:
            y_train.append([label_to_index[label] for label in lab])
        for lab in y_test_tmp:
            y_test.append([label_to_index[label] for label in lab])

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(all_label_list), y=all_label_list)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        print("Label to index: ", label_to_index)
        print("Class weights: ", class_weight_dict)

        # Binarize the labels for multi-label classification
        mlb = MultiLabelBinarizer()
        y_train = mlb.fit_transform(y_train)
        y_test = mlb.fit_transform(y_test)

        svm_classifier = OneVsRestClassifier(SVC(class_weight=class_weight_dict, max_iter=10000))
        lr_classifier = OneVsRestClassifier(LogisticRegression(class_weight=class_weight_dict, max_iter=10000))

        return X_train, y_train, X_test, y_test, svm_classifier, lr_classifier
    
    ## many inputs, one label
    elif dataset in inputs_label_sets:
        X_trains = train_data.loc[:, train_data.columns != 'label']
        print("X_trains.columns: ", X_trains.columns)
        X_tests = test_data.loc[:, test_data.columns != 'label']
        print("X_tests.columns: ", X_tests.columns)

        inputs = []
        for inp in X_trains.columns:
            inp = X_trains[inp]
            inp = [preprocess(text) for text in inp]
            inp_emb = np.array([get_sentence_embedding(text, fasttext_embeddings) for text in tqdm(inp, desc="Getting train embeds")])
            inputs.append(inp_emb.copy())
        inputs = np.array(inputs)
        X_train = np.concatenate(inputs,axis=1)

        inputs = []
        for inp in X_tests.columns:
            inp = X_tests[inp]
            inp = [preprocess(text) for text in inp]
            inp_emb = np.array([get_sentence_embedding(text, fasttext_embeddings) for text in tqdm(inp, desc="Getting train embeds")])
            inputs.append(inp_emb.copy())
        inputs = np.array(inputs)
        X_test = np.concatenate(inputs,axis=1)

        y_train, y_test = train_data['label'].tolist(), test_data['label'].tolist()
        # Convert labels to numerical format
        unique_labels = list(set(y_train).union(set(y_test)))
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        y_train = [label_to_index[label] for label in y_train]
        y_test = [label_to_index[label] for label in y_test]

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        print("Label to index: ", label_to_index)
        print("Class weights: ", class_weight_dict)

        svm_classifier = SVC(class_weight=class_weight_dict, max_iter=10000)
        lr_classifier = LogisticRegression(class_weight=class_weight_dict, max_iter=10000)

        return X_train, y_train, X_test, y_test, svm_classifier, lr_classifier

    ## Unidentifiable dataset
    else:
        print("Unknown data: ", dataset)
        return -1
    
X_train, y_train, X_test, y_test, svm_classifier, lr_classifier = get_data_ready(train_path, test_path)

# Train and evaluate SVM
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
class_report = classification_report(y_test, svm_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print(f"SVM Report\n {class_report}")

# Train and evaluate Logistic Regression
lr_classifier.fit(X_train, y_train)
lr_predictions = lr_classifier.predict(X_test)
class_report = classification_report(y_test, lr_predictions)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}")
print(f"LR Report\n {class_report}")