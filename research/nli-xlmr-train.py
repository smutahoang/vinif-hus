from transformers import AdamW
import torch.nn as nn
from tqdm import tqdm_notebook, tqdm
import random
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import pickle
import datetime
import os, sys

# Load the tokenizer.
xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

xlmr_config = XLMRobertaConfig.from_pretrained(
    "xlm-roberta-base", from_tf=False, num_labels=3, output_hidden_states=False,
)
xlmr_model = XLMRobertaForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    config=xlmr_config
)

device = torch.device('cuda')

#xlmr_model = torch.nn.DataParallel(xlmr_model)
xlmr_model.to(device)

train_inputs = pickle.load(open('/home/hoang/qnt/data_nli/nli-train_inputs_full.pkl', 'rb'))
val_inputs = pickle.load(open('/home/hoang/qnt/data_nli/nli-val_inputs_full.pkl', 'rb'))
train_masks = pickle.load(open('/home/hoang/qnt/data_nli/nli-train_masks_full.pkl', 'rb'))
val_masks = pickle.load(open('/home/hoang/qnt/data_nli/nli-val_masks_full.pkl', 'rb'))
train_labels = pickle.load(open('/home/hoang/qnt/data_nli/nli-train_labels_full.pkl', 'rb'))
val_labels = pickle.load(open('/home/hoang/qnt/data_nli/nli-val_labels_full.pkl', 'rb'))
label_to_idx = pickle.load(open('/home/hoang/qnt/data_nli/nli-label_to_idx_full.pkl', 'rb'))
idx_to_label = pickle.load(open('/home/hoang/qnt/data_nli/nli-idx_to_label_full.pkl', 'rb'))


train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=256)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=256)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    F1_score = f1_score(pred_flat, labels_flat, average='macro')

    return accuracy_score(pred_flat, labels_flat), F1_score

def save(model, optimizer, path):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


epochs = 100

w = open('/home/hoang/qnt/nli-xlmr_full_100epoch_detail.out','at',encoding='utf-8')
w.write("-------------------------------- "+str(epochs)+' epochs----------------------\n')
w.write(str(datetime.datetime.now())+'\n')
w.close()
param_optimizer = list(xlmr_model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)

pre_model_name = ''

pre_f1_score = 0

for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    w = open('/home/hoang/qnt/nli-xlmr_full_100epoch_detail.out','at',encoding='utf-8')
    w.write(str(datetime.datetime.now())+'\n')
    w.write('======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, epochs))
    w.close()


    w = open('/home/hoang/qnt/nli-xlmr_full_100epoch.out','at',encoding='utf-8')
    w.write(str(datetime.datetime.now())+'\n')
    w.write('======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, epochs))
    w.close()
    
    print('Training...')

    total_loss = 0
    xlmr_model.train()
    train_accuracy = 0
    nb_train_steps = 0
    train_f1 = 0

    for step, batch in tqdm(enumerate(train_dataloader)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        xlmr_model.zero_grad()
        outputs = xlmr_model(b_input_ids,
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
        torch.nn.utils.clip_grad_norm_(xlmr_model.parameters(), 1.0)
        optimizer.step()
		
        w = open('/home/hoang/qnt/nli-xlmr_full_100epoch_detail.out','at',encoding='utf-8')
        w.write(' ------ Done step'+str(step)+ ' |'+str(datetime.datetime.now())+'|\n')
       
        w.close()

    avg_train_loss = total_loss / len(train_dataloader)
    print(" Accuracy: {0:.4f}".format(train_accuracy/nb_train_steps))
    print(" F1 score: {0:.4f}".format(train_f1/nb_train_steps))
    print(" Average training loss: {0:.4f}".format(avg_train_loss))
    

    print("Running Validation...")

    w = open('/home/hoang/qnt/nli-xlmr_full_100epoch_detail.out','at',encoding='utf-8')
    w.write(" Accuracy: {0:.4f}\n".format(train_accuracy/nb_train_steps))
    w.write(" F1 score: {0:.4f}\n".format(train_f1/nb_train_steps))
    w.write(" Average training loss: {0:.4f}\n".format(avg_train_loss))
    w.write("Running Validation...\n")
    w.close()

    w = open('/home/hoang/qnt/nli-xlmr_full_100epoch.out','at',encoding='utf-8')
    w.write(" Accuracy: {0:.4f}\n".format(train_accuracy/nb_train_steps))
    w.write(" F1 score: {0:.4f}\n".format(train_f1/nb_train_steps))
    w.write(" Average training loss: {0:.4f}\n".format(avg_train_loss))
    w.write("Running Validation...\n")
    w.close()

    xlmr_model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    eval_f1 = 0
    for batch in tqdm(val_dataloader):

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = xlmr_model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy, tmp_eval_f1 = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            eval_f1 += tmp_eval_f1
            nb_eval_steps += 1
    
    f1_score = eval_f1/nb_eval_steps
    if f1_score > 0.8 and f1_score > pre_f1_score:
        new_model_path = '/home/hoang/qnt/nli-xlmr_full_100epoch_save_at_'+str(epoch_i)+'.bk'
        save(xlmr_model, optimizer, new_model_path)

        if os.path.exists(pre_model_name):
            os.remove(pre_model_name)
        
        pre_model_name = new_model_path
        pre_f1_score = f1_score



    print(" Accuracy: {0:.4f}".format(eval_accuracy/nb_eval_steps))
    print(" F1 score: {0:.4f}".format(eval_f1/nb_eval_steps))
    w = open('/home/hoang/qnt/nli-xlmr_full_100epoch_detail.out','at',encoding='utf-8')
    w.write(" Accuracy: {0:.4f}\n".format(eval_accuracy/nb_eval_steps))
    w.write(" F1 score: {0:.4f}\n".format(eval_f1/nb_eval_steps))
    w.close()


    w = open('/home/hoang/qnt/nli-xlmr_full_100epoch.out','at',encoding='utf-8')
    w.write(" Accuracy: {0:.4f}\n".format(eval_accuracy/nb_eval_steps))
    w.write(" F1 score: {0:.4f}\n".format(eval_f1/nb_eval_steps))
    w.close()

print("Training complete!")
w = open('/home/hoang/qnt/nli-xlmr_full_100epoch_detail.out','at',encoding='utf-8')
w.write("Training complete! ")
w.write(str(datetime.datetime.now())+'\n')
w.close()


w = open('/home/hoang/qnt/nli-xlmr_full_100epoch.out','at',encoding='utf-8')
w.write("Training complete! ")
w.write(str(datetime.datetime.now())+'\n')
w.close()


save(xlmr_model, optimizer, '/home/hoang/qnt/NLI_XLMR_100Epoch_full.bk')
