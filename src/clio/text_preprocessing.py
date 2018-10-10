from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def keras_tokenizer(texts, num_words=None, mode=None, maxlen=None, sequences=False):
    """Preprocess text with Keras Tokenizer.

    Args:
        texts (list, str): Collections of documents to preprocess.
        num_words (int): Length of the vocabulary.
        mode (str): Can be "count" or "tfidf".

    Returns:
        encoded_docs (array, int | float): Can be Bag-of-Words or TF-IDF weight matrix, depending
                                            on "mode".

    """
    t = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~ ', num_words=num_words)
    t.fit_on_texts(texts)

    if sequences:
        seq = t.texts_to_sequences(texts)
        encoded_docs = pad_sequences(seq, maxlen=maxlen)
    else:
        encoded_docs = t.texts_to_matrix(texts, mode=mode)

    print('Documents count: {}'.format(t.document_count))
    print('Found %s unique tokens.' % len(t.word_index))
    print('Shape of encoded docs: {}'.format(encoded_docs.shape))

    return encoded_docs
