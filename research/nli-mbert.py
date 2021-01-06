from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from transformers import BertTokenizer, BertModel
from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import random
from tqdm import tqdm_notebook, tqdm
import torch.nn as nn

data = pickle.load(open('snli_1.0_translated.pkl', 'rb'))

i = 0
while i < len(data):
    if data[i]['label'] == '-':
        del data[i]
        if i > 0:
            i -= 1
    i += 1


def makeOrder(s1, s2):
    if s1 > s2:
        return s2, s1

    return s1, s2


xdata = []
labels = []

for i in range(len(data)):
    s1, s2 = makeOrder(data[i]['sentence1'], data[i]['sentence2'])
    t1, t2 = makeOrder(data[i]['sentence1_translated'],
                       data[i]['sentence2_translated'])
    ms1, ms2 = makeOrder(data[i]['sentence1_translated'], data[i]['sentence2'])
    mt1, mt2 = makeOrder(data[i]['sentence1'], data[i]['sentence2_translated'])

    xdata.append(s1+' [SEP] '+s2)
    labels.append(data[i]['label'])
    xdata.append(t1+' [SEP] '+t2)
    labels.append(data[i]['label'])
    xdata.append(ms1+' [SEP] '+ms2)
    labels.append(data[i]['label'])
    xdata.append(mt1+' [SEP] '+mt2)
    labels.append(data[i]['label'])
    if i % 10000 == 0:
        print(i, '/', len(data), end='\r')


label_to_idx = {value: key for key, value in enumerate(labelSet)}
idx_to_label = {value: key for key, value in label_to_idx.items()}

print(label_to_idx)
print(idx_to_label)


labels = [label_to_idx[labels[i]] for i in range(len(labels))]


train_text, test_text, train_label, test_label = train_test_split(
    xdata, labels, test_size=0.1)
train_text, test_text, train_label, test_label = train_test_split(
    test_text, test_label, test_size=0.1)


print('train size = ', len(train_text))
print('test size = ', len(test_text))


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


train_inputs = []
val_inputs = []
train_labels = torch.tensor(train_label)
val_labels = torch.tensor(test_label)
train_masks = []
val_masks = []

for i in range(len(train_text)):
    encoded_input = tokenizer(
        train_text[i], return_tensors='pt', padding="max_length", truncation=True, max_length=80)
    train_inputs.append(encoded_input['input_ids'])
    train_masks.append(encoded_input['attention_mask'])
    if i % (len(train_text)//10) == 0:
        print(i, '/', len(train_text))

for i in range(len(test_text)):
    encoded_input = tokenizer(
        test_text[i], return_tensors='pt', padding="max_length", truncation=True,  max_length=80)
    val_inputs.append(encoded_input['input_ids'])
    val_masks.append(encoded_input['attention_mask'])
    if i % (len(test_text)//10) == 0:
        print(i, '/', len(test_text))

train_inputs = torch.cat(train_inputs)
val_inputs = torch.cat(val_inputs)
train_masks = torch.cat(train_masks)
val_masks = torch.cat(val_masks)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=128)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=128)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    F1_score = f1_score(pred_flat, labels_flat, average='macro')

    return accuracy_score(pred_flat, labels_flat), F1_score


config = RobertaConfig.from_pretrained(
    "bert-base-multilingual-cased", from_tf=False, num_labels=len(labelSet), output_hidden_states=False,
)
BERT_NLI = RobertaForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    config=config
)
BERT_NLI.cuda()


device = 'cuda'
epochs = 20

param_optimizer = list(BERT_NLI.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)


for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_loss = 0
    BERT_NLI.train()
    train_accuracy = 0
    nb_train_steps = 0
    train_f1 = 0

    for step, batch in tqdm(enumerate(train_dataloader)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        BERT_NLI.zero_grad()
        outputs = BERT_NLI(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()

        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_train_accuracy, tmp_train_f1 = flat_accuracy(logits, label_ids)
        train_accuracy += tmp_train_accuracy
        train_f1 += tmp_train_f1
        nb_train_steps += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(BERT_NLI.parameters(), 1.0)
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(" Accuracy: {0:.4f}".format(train_accuracy/nb_train_steps))
    print(" F1 score: {0:.4f}".format(train_f1/nb_train_steps))
    print(" Average training loss: {0:.4f}".format(avg_train_loss))

    print("Running Validation...")
    BERT_NLI.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    eval_f1 = 0
    for batch in tqdm_notebook(val_dataloader):

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = BERT_NLI(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy, tmp_eval_f1 = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            eval_f1 += tmp_eval_f1
            nb_eval_steps += 1
    print(" Accuracy: {0:.4f}".format(eval_accuracy/nb_eval_steps))
    print(" F1 score: {0:.4f}".format(eval_f1/nb_eval_steps))
print("Training complete!")



def save(model, optimizer, path):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

save(BERT_NLI, optimizer,'NLI_20Epoch.bk')
