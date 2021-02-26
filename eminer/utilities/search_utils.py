from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import requests
from urllib.parse import urlencode
from selenium import webdriver
from fake_useragent import UserAgent


def _get_search_url(query, site, num=10, lang='en', search_engine="google"):
    """Get search url
    """
    print(query)
    print(site)
    print(lang)

    if search_engine.lower() == 'google':
        params_gg = {
            'lr': lang,
            'q': query.encode('utf8'),
            'num': num
        }
        if site is not None:
            params_gg['as_sitesearch'] = site
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

    if lang == 'vi':
        url = u'https://www.{}.com.vn/search?'.format(search_engine)
    elif lang == 'en':
        url = u'https://www.{}.com/search?'.format(search_engine)
    url += params

    return url


def _get_html(url):
    """Get html from search url
    """
    driver = webdriver.Firefox()
    driver.get(url)
    elem = driver.find_element_by_xpath("//*")
    html = elem.get_attribute("outerHTML")
    driver.close()
    return html
