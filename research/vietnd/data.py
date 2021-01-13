import pickle

import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_data(cfg, version="xnli"):
    """Loading xnli or snli data
    """
    assert version in ["xnli", "snli"], "Should be xnli or snli"
    try:
        if version == "xnli":
            test_xnli_raw = pd.read_csv(cfg.xnli + "/xnli.test.tsv",
                                        sep='\t',
                                        header=0,
                                        error_bad_lines=False)
            dev_xnli_raw = pd.read_csv(cfg.xnli + "/xnli.dev.tsv",
                                       sep='\t',
                                       header=0,
                                       error_bad_lines=False)
            return test_xnli_raw, dev_xnli_raw
        elif version == "snli":
            with open(cfg.snli + "/snli_1.0_translated.pkl", "rb") as f:
                snli_raw = pickle.load(f)
                return snli_raw
    except FileNotFoundError:
        print("Invalid path")


def xnli_process(xnli_raw_data):
    """
    """
    test_raw, dev_raw = xnli_raw_data
    raw = pd.concat((test_raw, dev_raw)).reset_index(drop=True)
    del test_raw, dev_raw
    rs = {'label': [],
          'en': [],
          'vi': []}
    pbar = tqdm(total=7500)
    for i in raw.loc[raw.language == 'en', ['pairID']].iterrows():
        rs['label'].append(raw.iloc[i[0], 1])
        rs['en'].append(raw.iloc[i[0], 6])
        rs['vi'].append(raw.loc[(raw.language == 'vi') & (raw.pairID == i[1].pairID), 'sentence2'].values[0])
        pbar.update(1)
    pbar.close()

    with open('en_vi_xnli.pkl', 'wb') as handle:
        pickle.dump(rs, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
                "label": self._get_label(self.content.iloc[idx, 0])}

    def __len__(self):
        return len(self.content)

    def _get_label(self, x):
        label = {'contradiction': 0,
                 'neutral': 1,
                 'entailment': 2,
                 }

        return label[x]
