def select(articles, entailment_label, top_k=5):
    """Select top_k relevant sentences given articles and label

    :param articles: list of articles
    :param top_k: number of selected sentences
    :param entailment_label:
    :return: top_k indices in the sentences list
    """
    rs = []
    for article in articles:
        rs.extend(articles['relevant_sentences'])

    return sorted(rs, key=lambda x: x["entailment"][entailment_label], reverse=True)[:top_k]
