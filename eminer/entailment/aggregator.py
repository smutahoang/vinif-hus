# to determine entailment label between a document and a post by aggregating the labels
# between the document's sentences and the post

def avg(sentences, post):
    """
    aggregating by simple averaging
    :param sentences:
    :param post:
    :return:
    """

    # TODO: to be implemented
    pass


def aggregate(sentences, method='avg'):
    """

    :param sentences:
    :param method:
    :return: either 'support', 'refuse', or 'not-determined'
    """
    if method == 'avg':
        return avg(sentences, post)
    else:
        # TODO: to be implemented
        pass
