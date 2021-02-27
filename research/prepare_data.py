import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# Load raw data
data = pickle.load(open('snli_1.0_translated.pkl','rb'))
data = [d for d in data if d['label']!='-']

# Make pairs order
def makeOrder(s1, s2):
  if s1 > s2:
    return s2, s1
  
  return s1, s2


# concate pair into single sentence
xdata = []
labels = []

for i in range(len(data)):
  s1, s2 = makeOrder(data[i]['sentence1'],data[i]['sentence2'])
  t1, t2 = makeOrder(data[i]['sentence1_translated'],data[i]['sentence2_translated'])
  ms1, ms2 = makeOrder(data[i]['sentence1_translated'],data[i]['sentence2'])
  mt1, mt2 = makeOrder(data[i]['sentence1'],data[i]['sentence2_translated'])

  xdata.append(s1+' [SEP] '+s2)
  labels.append(data[i]['label'])
  xdata.append(t1+' [SEP] '+t2)
  labels.append(data[i]['label'])
  xdata.append(ms1+' [SEP] '+ms2)
  labels.append(data[i]['label'])
  xdata.append(mt1+' [SEP] '+mt2)
  labels.append(data[i]['label'])
  if i % 10000 == 0:
    print(i,'/',len(data))

# Convert string label to idx (number)
labelSet = set(labels)

label_to_idx = {value:key for key, value in enumerate(labelSet)}
idx_to_label = {value:key for key, value in label_to_idx.items()}

print(label_to_idx)
print(idx_to_label)

labels = [label_to_idx[labels[i]] for i in range(len(labels))]


# split data to train and test set
train_text, test_text, train_label, test_label = train_test_split(xdata,labels,test_size = 0.1)

print('train size = ', len(train_text))
print('test size = ',len(test_text))

# init tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Convert raw data to number vector
train_inputs = []
val_inputs = []
train_labels = torch.tensor(train_label)
val_labels = torch.tensor(test_label)
train_masks = []
val_masks = []

for i in range(len(train_text)):
  encoded_input = tokenizer(train_text[i], return_tensors='pt',padding="max_length",truncation=True, max_length=80)
  train_inputs.append(encoded_input['input_ids'])
  train_masks.append(encoded_input['attention_mask'])
  if i % (len(train_text)//10) == 0:
    print(i,'/',len(train_text))

for i in range(len(test_text)):
  encoded_input = tokenizer(test_text[i], return_tensors='pt',padding="max_length",truncation=True,  max_length=80)
  val_inputs.append(encoded_input['input_ids'])
  val_masks.append(encoded_input['attention_mask'])
  if i % (len(test_text)//10) == 0:
    print(i,'/',len(test_text))

train_inputs = torch.cat(train_inputs)
val_inputs = torch.cat(val_inputs)
train_masks = torch.cat(train_masks)
val_masks = torch.cat(val_masks)

# dumple data to pickle files

pickle.dump(train_inputs,open('nli-train_inputs_full.pkl','wb'))
pickle.dump(val_inputs,open('nli-val_inputs_full.pkl','wb'))
pickle.dump(train_masks,open('nli-train_masks_full.pkl','wb'))
pickle.dump(val_masks,open('nli-val_masks_full.pkl','wb'))
pickle.dump(train_labels,open('nli-train_labels_full.pkl','wb'))
pickle.dump(val_labels,open('nli-val_labels_full.pkl','wb'))
pickle.dump(label_to_idx,open('nli-label_to_idx_full.pkl','wb'))
pickle.dump(idx_to_label,open('nli-idx_to_label_full.pkl','wb'))


