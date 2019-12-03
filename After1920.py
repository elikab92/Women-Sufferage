import warnings
warnings.filterwarnings('ignore')
from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp
import re


#text = "hello world \n hello nice world \n hi world \n"
file = open('post-1920-books.txt','rt')
text = file.read()
file.close()

def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
    return filter(None, re.split(token_delim + '|' + seq_delim, source_str))
counter = nlp.data.count_tokens(simple_tokenize(text))
print(sorted(counter.items())[:10])
print('elika')
#print(counter[0:12])

#create indices for each token
vocab = nlp.Vocab(counter)

#Create a word embedding instance
fasttext_simple = nlp.embedding.create('fasttext', source='wiki.simple')

#Attach it to vocab
vocab.set_embedding(fasttext_simple)

#Get weight for each vector
input_dim, output_dim = vocab.embedding.idx_to_vec.shape
layer = gluon.nn.Embedding(input_dim, output_dim)
layer.initialize()
layer.weight.set_data(vocab.embedding.idx_to_vec)

print(len(vocab))

#Define cosine similarity

def cos_sim(x, y):
    return nd.dot(x, y) / (nd.norm(x) * nd.norm(y))

#Normalize each row
def norm_vecs_by_row(x):
    return x / nd.sqrt(nd.sum(x * x, axis=1) + 1E-10).reshape((-1,1))

#Finding the k nearest words
def get_knn(vocab, k, word):
    word_vec = vocab.embedding[word].reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(vocab.embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_vec)
    indices = nd.topk(dot_prod.reshape((len(vocab), )), k=k+1, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    # Remove unknown and input tokens.
    return vocab.to_tokens(indices[1:])

print(get_knn(vocab, 20, 'woman'))





