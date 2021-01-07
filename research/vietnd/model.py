import os

from vncorenlp import VnCoreNLP
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoModel, AutoTokenizer

from utils import flatten, checkvncorenlp


# PhoBert large
model_phobert_large = AutoModel.from_pretrained("vinai/phobert-large",
                                                return_dict=True,
                                                output_hidden_states=True)
tokenizer_phobert_large = AutoTokenizer.from_pretrained("vinai/phobert-large",
                                                        use_fast=False)

# XLM-R large
tokenizer_xlmr_large = RobertaTokenizer.from_pretrained('roberta-large')
model_xlmr_large = RobertaModel.from_pretrained('roberta-large',
                                                return_dict=True,
                                                output_hidden_states=True)

# Freeze model
for param in model_phobert_large.base_model.parameters():
    param.requires_grad = False
for param in model_xlmr_large.base_model.parameters():
    param.requires_grad = False


class MyEnsemble(torch.nn.Module):
    """
    """
    def __init__(self, device):
        super(MyEnsemble, self).__init__()
        self.device = device
        self.linear1 = torch.nn.Linear(4096, 1024)
        self.linear2 = torch.nn.Linear(1024, 128)
        self.linear3 = torch.nn.Linear(128, 3)

        if checkvncorenlp(os.getcwd()):
            self.rdrsegmenter = VnCoreNLP(os.getcwd() + "/vncorenlp/VnCoreNLP-1.1.1.jar",
                                          annotators="wseg",
                                          max_heap_size='-Xmx500m')

    def forward(self, en, vi):
        vector_en = self._output_xlmr(en)
        vector_vi = self._output_phobert(vi)
        x = self._matching(vector_en, vector_vi)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

    def _output_phobert(self, text):
        """Output from phobert model
        """
        def batch_process(list_text):
            sent = map(lambda x: ' '.join(list(flatten(x))),
                       [self.rdrsegmenter.tokenize(i) for i in list_text])
            inputs = tokenizer_phobert_large(list(sent),
                                             return_tensors="pt",
                                             truncation=True,
                                             padding=True).to(self.device)
            return inputs

        if isinstance(text, str):
            sentences = self.rdrsegmenter.tokenize(text)
            inputs = tokenizer_phobert_large(' '.join(list(flatten(sentences))),
                                             return_tensors="pt",
                                             truncation=True,
                                             padding=True).to(self.device)
        elif isinstance(text, list):
            inputs = batch_process(text)

        with torch.no_grad():
            output_phobert = model_phobert_large(**inputs)

        return output_phobert.pooler_output

    def _output_xlmr(self, text):
        """Output from xlmr model
        """
        inputs = tokenizer_xlmr_large(text,
                                      return_tensors="pt",
                                      truncation=True,
                                      padding=True).to(self.device)
        outputs = model_xlmr_large(**inputs)

        return outputs.pooler_output

    def _matching(self, en, vi):
        """Heuristic matching based on https://arxiv.org/pdf/1512.08422.pdf
        Params
        ----
         - en (Tensor): encoder vector for vietnamese sentence
         - vi (Tensor): encoder vector for english sentence
        Returns
        ----
         -
        """
        return torch.cat((en, vi, torch.abs(en-vi), torch.mul(en, vi)), dim=-1)
