from vncorenlp import VnCoreNLP
import torch
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoModel, AutoTokenizer

from utils import flatten


class MyEnsemble(torch.nn.Module):
    """
    """
    def __init__(self, device):
        super(MyEnsemble, self).__init__()
        self.device = device
        self.linear1 = torch.nn.Linear(4096, 1024)
        self.linear2 = torch.nn.Linear(1024, 128)
        self.linear3 = torch.nn.Linear(128, 3)
        self.rdrsegmenter = VnCoreNLP("/kaggle/working/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

    def forward(self, en, vi):
        vector_en = self._output_xlmr(en)
        vector_vi = self._output_phobert(vi)
        x = self._matching(vector_en, vector_vi)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    def _phobert(self):
        """Pho-Bert (large)
        """
        self.model_phobert_large = AutoModel.from_pretrained("vinai/phobert-large",
                                                             return_dict=True,
                                                             output_hidden_states=True)
        self.tokenizer_phobert_large = AutoTokenizer.from_pretrained("vinai/phobert-large", use_fast=False)
        for param in self.model_phobert_large.base_model.parameters():
            param.requires_grad = False

    def _xlmr(self):
        # XLM-R (original)
        self.tokenizer_xlmr_large = RobertaTokenizer.from_pretrained('roberta-large')
        self.model_xlmr_large = RobertaModel.from_pretrained('roberta-large', return_dict=True, output_hidden_states=True)

        for param in self.model_xlmr_large.base_model.parameters():
            param.requires_grad = False

    def _output_phobert(self, text):
        """Output from phobert model
        """
        sentences = self.rdrsegmenter.tokenize(text)
        input_ids = torch.tensor(
            [self.tokenizer_phobert_large.encode(' '.join(list(flatten(sentences))))]
            )

        with torch.no_grad():
            output_phobert = self.model_phobert_large(input_ids.to(self.device))

        return output_phobert.pooler_output

    def _output_xlmr(self, text):
        """Output from xlmr model
        """
        inputs = self.tokenizer_xlmr_large(text, return_tensors="pt").to(self.device)
        outputs = self.model_xlmr_large(**inputs)

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
        return torch.cat((en, vi, torch.abs(en-vi), torch.mul(en, vi)), dim=1)
