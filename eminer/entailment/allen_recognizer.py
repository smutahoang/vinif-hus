# make use of google translate API and Allen NLP's toolbox for entailment recognition
# Allen NLP toolbox: https://demo.allennlp.org/textual-entailment
from allennlp.predictors.predictor import Predictor
import cld3


def google_translate(sentence, api_key=None):
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
    return cld3.get_language(text)[0]


def infer_entailment(sent1, sent2, api_key):
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
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/mnli-roberta-2020-07-29.tar.gz")
    params = {'premise': google_translate(sent1) if lang_detect(sent1) != 'en' else sent1,
              'hypothesis': google_translate(sent2) if lang_detect(sent2) != 'en' else sent2
              }

    rs = predictor.predict(**params)

    return dict(zip(["entailment", "contradiction", "neutral"], rs["prob"]))
