from document import search_engine
from newspaper import Article


def search_articles(keyword, sources, engine='Google', top_k=10):
    """
    make use of the search-engine to find news articles from sources that are relevant to keywords,
    and then extract the top-k links returned by the engine
    :param keyword: string, the keyword to search, e.g, "covid-19 cases"
    :param sources: list of strings, each is a source, i.e., a news channel, e.g., ["bbc.co.uk", "vnexpress.net"]
    :param engine: 'Google' or 'Bing', etc.
    :param top_k: number of top result to retrieve
    :return: list of top-k links return by the search engine
    """
    result = {}
    for source in sources:
        result[source] = [i.link for i in
                          search_engine.search(keyword, source, num=top_k, lang='en', engine=engine)]

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
    # return get_news(url)
    article = Article(url)
    article.download()
    article.parse()
    return {'title': article.title, 'paragraphs': article.text.split('\n'), 'images': list(article.images)}


def retrieve(keywords, sources, engine='google', top_k=10):
    """
    make use of the search-engine to retrieve news articles from sources that are relevant to keywords
    :param keywords: list of strings, the keyword to search, e.g, "covid-19 cases"
    :param sources: list of strings, each is a source, i.e., a news channel, e.g., ["bbc.co.uk", "vnexpress.net"]
    :param engine: 'Google' or 'Bing', etc.
    :param top_k: number of top result to retrieve for each keyword
    :return: dictionary, the extracted news article, including
        {
            "url": url to the article,
            "title": the title of the article,
            "paragraphs": list of string, each is a paragraph,
            "images": list of images included in the news
        }
    """
    urls = {}
    for keyword in keywords:
        urls[keyword] = search_articles(keyword, sources, engine=engine, top_k=top_k)
    all_urls = []
    for _, u in urls.items():
        for source in sources:
            all_urls.extend(urls[keyword][source])
    urls = set(all_urls)
    articles = [extract_news(url) for url in urls]
    return articles
