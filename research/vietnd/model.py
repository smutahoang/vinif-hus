import os

import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoModel, AutoTokenizer

from utils import flatten, check_vncorenlp


# PhoBert large
model_phobert_large = AutoModel.from_pretrained("vinai/phobert-base",
                                                return_dict=True,
                                                output_hidden_states=True)
tokenizer_phobert_large = AutoTokenizer.from_pretrained("vinai/phobert-base",
                                                        use_fast=False)

if check_vncorenlp(os.getcwd()):
    from vncorenlp import VnCoreNLP
    rdrsegmenter = VnCoreNLP("vncorenlp_src/VnCoreNLP-1.1.1.jar",
                             annotators="wseg",
                             max_heap_size='-Xmx500m')

# XLM-R large
tokenizer_xlmr_large = RobertaTokenizer.from_pretrained('roberta-base')
model_xlmr_large = RobertaModel.from_pretrained('roberta-base',
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
    def __init__(self, device=None, mode_cls=False, mode_mean_hidden_state=True, cfg=None):
        super(MyEnsemble, self).__init__()
        self.mode_cls = mode_cls
        self.mode_mean_hidden_state = mode_mean_hidden_state
        self.device = device
        self.cfg = cfg
        self.dense = torch.nn.Linear(768*3, 768*3)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.out_proj = torch.nn.Linear(768*3, 3)

    def forward(self, en, vi):
        vector_en = self._encode(self._output_xlmr, en)
        vector_vi = self._encode(self._output_phobert, vi)
        x = self._matching(vector_en, vector_vi)
        x = self.dropout(self.dense(x))
        x = torch.tanh(x)
        x = self.out_proj(x)

        return x

    def _output_phobert(self, text):
        """Output from phobert model
        """
        def batch_process(list_text):
            sent = map(lambda x: ' '.join(list(flatten(x))),
                       [rdrsegmenter.tokenize(i) for i in list_text])
            inputs = tokenizer_phobert_large(list(sent),
                                             return_tensors="pt",
                                             truncation=True,
                                             padding=True).to(self.device)
            return inputs

        if isinstance(text, str):
            sentences = rdrsegmenter.tokenize(text)
            inputs = tokenizer_phobert_large(' '.join(list(flatten(sentences))),
                                             return_tensors="pt",
                                             truncation=True,
                                             padding=True).to(self.device)
        elif isinstance(text, list):
            inputs = batch_process(text)

        outputs = model_phobert_large(**inputs)

        return inputs, outputs

    def _output_xlmr(self, text):
        """Output from xlmr model
        """
        inputs = tokenizer_xlmr_large(text,
                                      return_tensors="pt",
                                      truncation=True,
                                      padding=True).to(self.device)
        outputs = model_xlmr_large(**inputs)

        return inputs, outputs

    def _encode(self, model, text):
        """Mean pooling of hidden state layer
        """
        inputs, outs = model(text)
        if self.mode_cls:
            return outs["last_hidden_state"][:, 0, :]
        elif self.mode_mean_hidden_state:
            mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(outs["last_hidden_state"].size()).float()
            sum_embedding = torch.sum(outs["last_hidden_state"] * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-8)
            return sum_embedding / sum_mask

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
        return torch.cat((en, vi, torch.abs(en-vi)), dim=1)
