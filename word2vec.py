import warnings
warnings.filterwarnings('ignore')
from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp
import re


text = "hello world \n hello nice world \n hi world \n"

def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
    return filter(None, re.split(token_delim + '|' + seq_delim, source_str))
counter = nlp.data.count_tokens('prepared.txt')
print('elika')
print(counter[0:12])

vocab = nlp.Vocab(counter)

fasttext_simple = nlp.embedding.create('fasttext', source='wiki.simple')
vocab.set_embedding(fasttext_simple)
glove_840b = nlp.embedding.create('glove', source='glove.6B.50d')
vocab = nlp.Vocab(nlp.data.Counter(glove_840b.idx_to_token))
vocab.set_embedding(glove_840b)
def norm_vecs_by_row(x):
    return x / nd.sqrt(nd.sum(x * x, axis=1) + 1E-10).reshape((-1,1))

def get_knn(vocab, k, word):
    word_vec = vocab.embedding[word].reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(vocab.embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_vec)
    indices = nd.topk(dot_prod.reshape((len(vocab), )), k=k+1, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    # Remove unknown and input tokens.
    return vocab.to_tokens(indices[1:])

get_knn(vocab, 5, 'woman')



for words in vocab.idx_to_token:
    print(words)
