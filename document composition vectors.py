
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.corpora import Dictionary
from gensim.models import fasttext
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
from scipy.spatial.distance import cosine
#%%
class SCDV(object):

    def __init__(self, n_components=100, min_count=5, epochs=5):
        self.n_components = n_components
        self.min_count = min_count
        self.iter = epochs

    def __get_sentences_tokens(self, documents):
        _data = []
        for doc in documents:
            _doc = doc.lower()
            _sentences = sent_tokenize(_doc)
            for sent in _sentences:
                _data.append(word_tokenize(sent))
        return _data

    def fit(self, documents=None):
        self.documents = documents
        _resumes_words_list = self.__get_sentences_tokens(self.documents)

        self.model_tfidf = TfidfVectorizer()
        self.model_tfidf.fit(self.documents)

        self.model_word2vec = fasttext.FastText(_resumes_words_list,
                                                negative=5,
                                                workers=4,
                                                iter=self.iter,
                                                min_count=self.min_count)
        self.word_vectors = self.model_word2vec.wv.syn0

        self.model_cluster = GaussianMixture(n_components=self.n_components)
        self.model_cluster.fit(self.word_vectors)

    def get_document_vector(self, document):

        if type(document) == type("str"):
            document = word_tokenize(document.lower())

        doc1 = [document]

        # compute word vectors for words in the document
        _doc_topic_wv = []
        for i in doc1[0]:
            try:
                val = self.model_word2vec.wv[i]
            except KeyError:
                val = numpy.zeros_like(self.model_word2vec.wv.syn0[0])
            _doc_topic_wv.append(val)

        # compute topic probabilities for each word vector
        _doc_topic_probs = self.model_cluster.predict_proba(_doc_topic_wv)


        # multiply each word vector repeated n-times with topic probabilities.
        # n: number of topics
        _doc_topic_wv = numpy.array(
            [numpy.repeat(
                [_doc_topic_wv[i]],
                len(_doc_topic_probs[i]),
                axis=0
            )
                for i in range(len(_doc_topic_wv))
            ]
        )
        _doc_topic_probs = numpy.array(_doc_topic_probs)
        _doc_topic_probs = _doc_topic_probs.reshape(
            [
                _doc_topic_probs.shape[0],
                _doc_topic_probs.shape[1], 1
            ]
        )
        _prob_mul_matrix = numpy.multiply(_doc_topic_wv, _doc_topic_probs)

        # get the tfidf weights of each word in te document
        _indexes_weights_tfidf = self.__get_tfidf__([" ".join(doc1[0])])
        _indexes_weights_tfidf = _indexes_weights_tfidf.reshape(
            [_indexes_weights_tfidf.shape[0], 1])
        _indexes_weights_tfidf = numpy.repeat(
            _indexes_weights_tfidf, repeats=self.n_components, axis=1)
        _indexes_weights_tfidf = _indexes_weights_tfidf.reshape(
            [_indexes_weights_tfidf.shape[0],
             _indexes_weights_tfidf.shape[1],
             1
             ]
        )

        # mutiply the tfidf of each word with the above _prob_mul_matrix
        # to get the weighted word vector probabilit matrix
        _prob_mul_matrix = numpy.multiply(
            _prob_mul_matrix, _indexes_weights_tfidf)

        # sum element along rows i.e. all words to get the final vectors and reshape 
        _prob_mul_matrix = numpy.sum(_prob_mul_matrix, axis=0)
        _prob_mul_matrix = _prob_mul_matrix.reshape([-1, 1])

        # increase the sparcity by reducing the less than 95% values to 0
#         _prob_mul_matrix[_prob_mul_matrix < numpy.percentile(_prob_mul_matrix,5)] = 0
        return _prob_mul_matrix

    def __get_tfidf__(self,document):
        _transformed = self.model_tfidf.transform(document)
        _to_return = []
        for word in word_tokenize(document[0]):
            _index = self.model_tfidf.vocabulary_.get(word)
            if _index is None:
                _to_return.append(0)
            else:
                _val = _transformed[0,_index]
                _to_return.append(_val)
        return numpy.array(_to_return)

    def get_ranking(self, documents, query):
        docs = []
        for i in documents:
            docs.append(self.get_document_vector(i))
        q = []
        for i in query:
            q.append(self.get_document_vector(i))
        sims = []
        for i in q:
            for j in docs:
                sims.append(cosine(j, i))
        return sims


