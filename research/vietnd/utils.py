import os
import pip
from collections import Iterable


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
    """To check RDRSegmenter from VnCoreNLP exists or not
    """
    try:
        import vncorenlp
        print("module 'vncorenlp' is installed")
    except ModuleNotFoundError:
        print("Installing vncorenlp module")
        pip.main(['install', 'vncorenlp'])

    if not os.path.exists(os.path.join(path, 'vncorenlp')):
        raise Exception("You must download VnCoreNLP-1.1.1.jar & its word segmentation component and put it in the same working folder with model")
