# make use of google translate API and Allen NLP's toolbox for entailment recognition
# Allen NLP toolbox: https://demo.allennlp.org/textual-entailment

def google_translate(sentence, api_key):
    """
    to translate a sentence in Vietnamese into English

    :param sentence: string, the sentence to be translated
    :param api_key: string, the key to call Google Translate API
    :return: string the sentence translated into English
    """
    # TODO: to be implemented
    pass


def lang_detect(text):
    """
    make use of langdetect to determine the language of the text
    :param text: string
    :return: either 'en', 'vi', or 'other' for English, Vietnamese, and Others respectively
    """
    # TODO: to be implemented
    pass


def infer_entailment(sentence1, sentence2, api_key):
    """
    infer entailment between sentence1 and sentence2, translate into English if needed
    :param sentence1: string, the first sentence
    :param sentence2: string, the second sentence
    :param api_key: string, the key to call Google Translate API
    :return: dictionary, keys are entailment labels, values are the corresponding probabilities, including
        {
            'entailment': entailment probability,
            'contradict': contradict probability,
            'neutral': neutral  probability
        }
    """
    # TODO: to be implemented

    pass
