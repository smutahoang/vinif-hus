from keywords import yake, snlp, bert
from document import news_retriever
from entailment import allen_recognizer
from entailment import aggregator
from summarization import sentence_selector
from utilities import nlp

information_sources = ['cnn.com',
                       'nytimes.com',
                       'washingtonpost.com',
                       'vnexpress.net',
                       'dantri.com',
                       'zing.vn']


def extract_keywords(post, method):
    """
    extract keyword from post
    :param post:
    :param method:
    :return:
    """
    if method == 'yake':
        return yake.get_keywords(post)
    elif method == 'snlp':
        return snlp.get_keywords(post)
    elif method == 'bert':
        return bert.get_keywords(post)
    else:
        raise NotImplementedError("must be yake or snlp or bert")


def retrieve_articles(keywords, sources):
    """
    to retrieve news articles relevant to keywords
    :param keywords:
    :param sources:
    :return:
    """
    articles = news_retriever.retrieve(keywords, sources)
    return articles


def infer_entailment(articles, post, method='allen'):
    """

    :param articles:
    :param post:
    :param method:
    :return:
    """
    recognizer = None
    if method == 'allen':
        recognizer = allen_recognizer
    else:
        # TODO: to be implemented
        pass
    for a in articles:
        relevant_sentences = nlp.get_relevant_sentence(a['paragraphs'], post)
        a['relevant_sentences'] = [{'text': s,
                                    'entailment': recognizer.infer_entailment(s, post)} for s in
                                   relevant_sentences]
        a['entailment'] = aggregator.aggregate(a['relevant_sentences'])

    return articles


def summarize(articles):
    """

    :param articles:
    :return:
    """
    supporting_articles = [a for a in articles if a['entailment'] == 'support']
    refusing_articles = [a for a in articles if a['entailment'] == 'refuse']
    evidences = {'support': sentence_selector.select(supporting_articles, entailment_label='support'),
                 'refuse': sentence_selector.select(refusing_articles, entailment_label='refuse')}
    return evidences


def find_evidence(post, keyword_method, entail_method):
    """

    :param post:
    :return:
    """
    keywords = extract_keywords(post, method=keyword_method)
    articles = retrieve_articles(keywords, information_sources)
    return summarize(infer_entailment(articles, post, method=entail_method))
