# to determine entailment label between a document and a post by aggregating the labels
# between the document's sentences and the post
import numpy as np


def avg(article):
    """
    aggregating by simple averaging
    :param article:
    :param post:
    :return: dict
        {
            "entailment": mean of prob. value over all relevant sentences
            "neutral": ~
            "contradiction": ~
        }
    """
    rs = []
    for k in article["releven_sentences"]:
        rs.append(list(k["entailment"].values()))
    rs = np.array(rs).mean(axis=0).round(3).tolist()
    return dict(zip(["entailment", "neutral", "contradiction"], rs))


def aggregate(sentences, method='avg'):
    """

    :param sentences:
    :param method:
    :return: either 'support', 'refuse', or 'not-determined'
    """
    if method == 'avg':
        return max(avg(sentences).items(), key=lambda x: x[1])[0]
    else:
        # TODO: to be implemented
        pass
