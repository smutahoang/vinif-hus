import os
import pip
from collections import Iterable

import wget


def flatten(lis):
    """Flatten nested list
    """
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def check_vncorenlp(path):
    """To check RDRSegmenter from VnCoreNLP exists or not. If not, create folder and download it
    """
    try:
        import vncorenlp
        print("module 'vncorenlp' is installed")
    except ModuleNotFoundError:
        print("Installing vncorenlp module")
        pip.main(['install', 'vncorenlp'])

    if not os.path.exists('vncorenlp_src'):
        print("Create vncorenlp_src folder")
        os.makedirs("vncorenlp_src/models/wordsegmenter")
        print("Start downloading VnCoreNLP-1.1.1.jar & its word segmentation component")
        url1 = 'https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar'
        url2 = 'https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab'
        url3 = 'https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr'
        wget.download(url1, 'vncorenlp_src/')
        wget.download(url2, 'vncorenlp_src/models/wordsegmenter/')
        wget.download(url3, 'vncorenlp_src/models/wordsegmenter/')

    return True
