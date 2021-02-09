from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import requests
from urllib.parse import urlencode
from fake_useragent import UserAgent


def _get_search_url(query, site, num=10, lang='en', search_engine="google"):
    """Get search url
    """
    if search_engine.lower() == 'google':
        params_gg = {
            'as_sitesearch': site if site else None,
            'lr': lang,
            'q': query.encode('utf8'),
            'num': num
        }
        params = urlencode(params_gg)
    elif search_engine.lower() == 'bing':
        if site is not None:
            query = r'site:{} {}'.format(site, query)
        params_bi = {
            'language': lang,
            'q': query.encode('utf8'),
            'count': num
        }
        params = urlencode(params_bi)

    if lang == 'vn':
        url = u'https://www.{}.com.vn/search?'.format(search_engine)
    elif lang == 'en':
        url = u'https://www.{}.com/search?'.format(search_engine)
    url += params

    return url


def _get_html(url):
    """Get html from search url
    """
    # ua = UserAgent()
    # header = ua.random
    header = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0"}

    try:
        r = requests.get(url, headers=header)
        return r.content
    except Exception as e:
        print("Error accessing:", url)
        print(e)
        return None
