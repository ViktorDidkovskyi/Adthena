
from blingfire import *
from unidecode import unidecode

from sklearn.base import TransformerMixin


class CleanTextTransformer(TransformerMixin):
    """ Class to clean the terms
    """
    def transform(self, X, **transform_params):
        return [text_to_words(clean_text(text)) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


def clean_text(text):
    # remove of newlines
    text = str(text)
    text = text.strip().replace("\n", " ").replace("\r", " ").replace('&', 'and')
    # replace HTML symbols
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    # convert to lowercase
    text = text.lower()

    return unidecode(text)


def text_to_words(s: str):
    """convert text to word"""

    # get the UTF-8 bytes
    s_bytes = s.encode("utf-8")

    # allocate the output buffer
    o_bytes = create_string_buffer(len(s_bytes) * 3)
    o_bytes_count = len(o_bytes)

    # identify paragraphs
    o_len = blingfire.TextToWords(c_char_p(s_bytes), c_int(len(s_bytes)), byref(o_bytes), c_int(o_bytes_count))

    # check if no error has happened
    if -1 == o_len or o_len > o_bytes_count:
        return ''

    # compute the unicode string from the UTF-8 bytes
    return o_bytes.value.decode('utf-8')

