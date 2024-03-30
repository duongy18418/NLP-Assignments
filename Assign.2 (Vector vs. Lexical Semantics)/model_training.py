import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("brown")
from nltk.corpus import brown
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
import pandas

simlex_999 = pandas.read_table("SimLex-999.txt")
corpus = brown.sents()

w2v_model_1 = Word2Vec(corpus, min_count=1, window=1, vector_size=10, epochs=1000, sg=1)
w2v_model_1 = w2v_model_1.wv
w2v_model_1.save("./Models/w2v_model_1.kv")

w2v_model_2 = Word2Vec(corpus, min_count=1, window=2, vector_size=50, epochs=1000)
w2v_model_2 = w2v_model_2.wv
w2v_model_2.save("./Models/w2v_model_2.kv")

w2v_model_5 = Word2Vec(corpus, min_count=1, window=5, vector_size=100, epochs=1000)
w2v_model_5 = w2v_model_5.wv
w2v_model_5.save("./Models/w2v_model_5.kv")

w2v_model_10 = Word2Vec(corpus, min_count=1, window=10, vector_size=300, epochs=1000)
w2v_model_10 = w2v_model_10.wv
w2v_model_10.save("./Models/w2v_model_10.kv")