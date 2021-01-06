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
