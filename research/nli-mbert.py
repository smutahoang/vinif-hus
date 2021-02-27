import numpy as np
import torch
import pickle
from transformers import BertTokenizer
import os
import gdown
from transformers import RobertaForSequenceClassification, RobertaConfig

class NliMBertClassifier:

    def __init__(self, trained_model_path: str,idx_to_label:dict, device='cpu') -> None:

        if not os.path.exists(trained_model_path):
            print('Model does not exist -> downloads default model...')
            url = 'https://drive.google.com/uc?id=190d73Qm4vB6_lu7yztsF7OH-iZtY-Ayo'
            output = 'NLI_MBERT_120Epoch_full.bk'
            gdown.download(url, output, quiet=False)
            trained_model_path = 'NLI_MBERT_120Epoch_full.bk'

        if not idx_to_label:
            print('download default labels map...')
            url = 'https://drive.google.com/uc?id=1-8UglnyNlIrhSFWUKuntwxoxVkaiOl6Z'
            output = 'nli-idx_to_label_full.pkl'
            gdown.download(url, output, quiet=False)

            idx_to_label = pickle.load(open(r"nli-idx_to_label_full.pkl", 'rb'))

        self.device = device
        self.idx_to_label = idx_to_label

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')



        print('Load trained model to device..')
        trained_model = torch.load(trained_model_path, map_location=device)
        
        config = RobertaConfig.from_pretrained(
            "bert-base-multilingual-cased", from_tf=False, num_labels=len(idx_to_label), output_hidden_states=False,
        )
        
        self.model = RobertaForSequenceClassification(config=config)
        self.model.load_state_dict(trained_model['model_state_dict'])
        #self.model.to(self.device)
        self.model.eval
        print('Load done !')


    def make_order(self, s1, s2):
        if s1 > s2:
            return s2, s1
        
        return s1, s2

    def make_input_data(self, pairs):
        inputs = []
        input_ids = []
        input_masks = []
        for pair in pairs:
            s1, s2 = self.make_order(pair[0],pair[1])
            text = s1 +' SEP] '+s2
            inputs.append(text)
            encoded_input = self.tokenizer(text, return_tensors='pt',padding="max_length",truncation=True, max_length=80)
            input_ids.append(encoded_input['input_ids'])
            input_masks.append(encoded_input['attention_mask'])

        input_ids = torch.cat(input_ids)
        input_masks = torch.cat(input_masks)
        
        return inputs, input_ids, input_masks


    def classify(self, pairs):

        inputs, input_ids, input_masks = self.make_input_data(pairs)

        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=None, attention_mask=input_masks)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        #print(logits.shape)
        result = [self.idx_to_label[idx] for idx in pred_flat]

        return result



if __name__ == '__main__':


    data = pickle.load(open(r'/home/hoang/qnt/snli_1.0_translated.pkl','rb'))
    data = [d for d in data if d['label'] !='-']

    

    idx_to_label = pickle.load(open(r"/home/hoang/qnt/data_nli/nli-idx_to_label_full.pkl", 'rb'))

    nli_xlmr_classifier = NliXlmrClassifier(r'/home/hoang/qnt/NLI_MBERT_120Epoch_full.bk',idx_to_label,device='cuda')
    # nli_xlmr_classifier = NliMBertClassifier(r'/home/hoang/qnt/NLI_XLMR_100Epoch_full.bk',idx_to_label,device='cpu')
    makeOrder = nli_xlmr_classifier.make_order
    xdata = []
    labels = []

    for i in range(10):
        s1, s2 = makeOrder(data[i]['sentence1'],data[i]['sentence2'])
        t1, t2 = makeOrder(data[i]['sentence1_translated'],data[i]['sentence2_translated'])
        ms1, ms2 = makeOrder(data[i]['sentence1_translated'],data[i]['sentence2'])
        mt1, mt2 = makeOrder(data[i]['sentence1'],data[i]['sentence2_translated'])

        xdata.append((s1,s2))
        labels.append(data[i]['label'])
        xdata.append((t1,t2))
        labels.append(data[i]['label'])
        xdata.append((ms1,ms2))
        labels.append(data[i]['label'])
        xdata.append((mt1,mt2))
        labels.append(data[i]['label'])

    result = nli_xlmr_classifier.classify(xdata)

    for i in range(len(xdata)):
        print(xdata[i])
        print('Gold [%s] vs Predict [%s] => [%r]' % (labels[i],result[i], labels[i]==result[i]))
