from search_engine import search


def search_articles(keyword, sources, search_engine='Google', top_k=10):
    """
    make use of the search-engine to find news articles from sources that are relevant to keywords,
    and then extract the top-k links returned by the engine
    :param keyword: string, the keyword to search, e.g, "covid-19 cases"
    :param sources: list of strings, each is a source, i.e., a news channel, e.g., ["bbc.co.uk", "vnexpress.net"]
    :param search_engine: 'Google' or 'Bing', etc.
    :param top_k: number of top result to retrieve
    :return: list of top-k links return by the search engine
    """
    result = {}
    for source in sources:
        result[source] = [i.link for i in search(keyword, source, top_k, search_engine)]

    return result


def extract_news(url):
    """
    download and extract the news article in the url
    :param url: string, link to a news article
        e.g., "https://vnexpress.net/ca-duong-tinh-sau-xuat-vien-lai-am-tinh-4192453.html"
    :return: dictionary, the extracted news article, including
        {
            "title": the title of the article
            "paragraphs": list of string, each is a paragraph
            "images": list of images included in the news
        }
    """
    # TODO: to be implemented
    pass


def retrieve(keywords, sources, search_engine='Google', top_k=10):
    """
    make use of the search-engine to retrieve news articles from sources that are relevant to keywords
    :param keywords: list of strings, the keyword to search, e.g, "covid-19 cases"
    :param sources: list of strings, each is a source, i.e., a news channel, e.g., ["bbc.co.uk", "vnexpress.net"]
    :param search_engine: 'Google' or 'Bing', etc.
    :param top_k: number of top result to retrieve for each keyword
    :return: dictionary, the extracted news article, including
        {
            "url": url to the article,
            "title": the title of the article,
            "paragraphs": list of string, each is a paragraph,
            "images": list of images included in the news
        }
    """
    urls = []
    for keyword in keywords:
        urls.append(search_articles(keyword, sources, search_engine=search_engine, top_k=top_k))
    urls = set(urls)
    articles = [extract_news(url) for url in urls]

    return articles
