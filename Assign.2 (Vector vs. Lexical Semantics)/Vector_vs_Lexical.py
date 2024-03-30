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
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec, KeyedVectors
import pandas
from sklearn.metrics import ndcg_score
import numpy
from matplotlib import pyplot

simlex_999 = pandas.read_table("./Assign.2 (Vector vs. Lexical Semantics)/SimLex-999.txt")
corpus = brown.sents()

w2v_model_1 = KeyedVectors.load('./Assign.2 (Vector vs. Lexical Semantics)/Models/w2v_model_1.kv')
w2v_model_2 = KeyedVectors.load('./Assign.2 (Vector vs. Lexical Semantics)/Models/w2v_model_2.kv')
w2v_model_5 = KeyedVectors.load('./Assign.2 (Vector vs. Lexical Semantics)/Models/w2v_model_5.kv')
w2v_model_10 = KeyedVectors.load('./Assign.2 (Vector vs. Lexical Semantics)/Models/w2v_model_10.kv')

token_corpus = pandas.DataFrame(columns=['word1', 'word2', 'cosine similarities'])
token_corpus_2 = pandas.DataFrame(columns=['word1', 'word2', 'cosine similarities'])
token_corpus_5 = pandas.DataFrame(columns=['word1', 'word2', 'cosine similarities'])
token_corpus_10 = pandas.DataFrame(columns=['word1', 'word2', 'cosine similarities'])


def vector_calculations(model, data):
    for i in simlex_999.index:
        word1 = simlex_999['word1'][i]
        word2 = simlex_999['word2'][i]
        if word1 in model and word2 in model:
            cosine = model.similarity(word1, word2)
            temp = (word1, word2, cosine)
            data.loc[len(data)] = temp
                
    data = data.groupby('word1', sort=False)['cosine similarities'].apply(list).to_frame()
    data.reset_index(inplace=True)
    data.index = numpy.arange(1, len(data)+1)

    temp_df = simlex_999.groupby('word1', sort=False)['SimLex999'].apply(list).to_frame()
    temp_df.reset_index(inplace=True)
    temp_df.index = numpy.arange(1, len(temp_df)+1)

    data = pandas.merge(data, temp_df, on='word1')

    data['nDCG Score'] = ""

    for i in data.index:
        c = [data['cosine similarities'][i]]
        g = [data['SimLex999'][i]]
        
        if len(data['cosine similarities'][i]) > 1:
            if len(data['cosine similarities'][i]) == len(data['SimLex999'][i]):
                score = ndcg_score(g, c)
                data['nDCG Score'][i] = score

    data['nDCG Score'].replace('', numpy.nan, inplace=True)
    data.dropna(subset=['nDCG Score'], inplace = True)
    data.set_index('word1', inplace=True)
    adv = data['nDCG Score'].mean(axis=0)
    return adv

adv_1 = vector_calculations(w2v_model_1, token_corpus)
adv_2 = vector_calculations(w2v_model_2, token_corpus_2)
adv_5 = vector_calculations(w2v_model_5, token_corpus_5)
adv_10 = vector_calculations(w2v_model_10, token_corpus_10)

data = {'Models': ['w2v_model_1', 'w2v_model_2', 'w2v_model_5', 'w2v_model_10'],
        'Adv nDCG Score': [adv_1, adv_2, adv_5, adv_10]}

ndcg_adverage = pandas.DataFrame(data)
ndcg_adverage.set_index('Models', inplace=True)
print(ndcg_adverage)

ndcg_adverage.plot(kind='bar', figsize=(10,5))
pyplot.title("Word2Vec Models Adverage nDCG Score")
pyplot.xlabel('Models')
pyplot.ylabel('nDCG Score')
pyplot.show()