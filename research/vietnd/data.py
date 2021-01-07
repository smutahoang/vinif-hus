import glob
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset


PATH = "/kaggle/input"


def load_data(version="xnli"):
    """Loading xnli or snli data
     - xnli data from https://www.kaggle.com/mzr2017/xnli-data
    """
    try:
        if version == "xnli":
            test_xnli_raw = pd.read_csv(PATH + "/xnli-data/xnli/test.tsv", sep='\t', header=0, error_bad_lines=False)
            dev_xnli_raw = pd.read_csv(PATH + "/xnli-data/xnli/dev.tsv", sep='\t', header=0, error_bad_lines=False)
            return test_xnli_raw, dev_xnli_raw
        elif version == "snli":
            with open("/kaggle/input/translated-snli/snli_1.0_translated.pkl", "rb") as f:
                snli_raw = pickle.load(f)
                return snli_raw
    except:
        print("Should be xnli or snli")


def xnli_process(xnli_raw_data):
    """
    """
    test_raw, dev_raw = xnli_raw_data
    # Get all of vietnamese sentence pairs from xnli
    vi = pd.concat([test_raw.loc[test_raw.language == 'vi'],
                    dev_raw.loc[dev_raw.language == 'vi']]).reset_index(drop=True)
    # Get all of en sentence pairs from xnli
    en = pd.concat([test_raw.loc[test_raw.language == 'en'],
                    dev_raw.loc[dev_raw.language == 'en']]).reset_index(drop=True)

    en_vi_xnli = pd.DataFrame({"label": vi.gold_label,
                               "premise": en.sentence1,
                               "hypothesis": vi.sentence2})

    return en_vi_xnli


def snli_process():
    """
    """
    pass


class xnliDataset(Dataset):
    """
    """
    def __init__(self, df):
        self.content = df

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {"en": self.content.iloc[idx, 1],
                "vi": self.content.iloc[idx, 2],
                "label": self.content.iloc[idx, 0]}

    def __len__(self):
        return len(self.content)
