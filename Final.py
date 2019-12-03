import warnings
warnings.filterwarnings('ignore')
from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp
import re
from monkeylearn import MonkeyLearn
import json
import os
import sys

result = None
fileName = "result-after1920.json" if sys.argv[1]=="after" else "result-before1920.json"
if not os.path.isfile(fileName):
    #text = "hello world \n hello nice world \n hi world \n"
    file = open('post-1920-books.txt' if sys.argv[1]=="after" else "before-1920-books.txt" ,'rt')
    text = file.read()
    file.close()

    def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
        return filter(None, re.split(token_delim + '|' + seq_delim, source_str))
    counter = nlp.data.count_tokens(simple_tokenize(text))
    print(sorted(counter.items())[:10])
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


    #Sentiment Analysis
    ml = MonkeyLearn('174a0d96f0e699a4d58a39a440205802710768f5')
    data = get_knn(vocab, 20, 'woman')
        #['The restaurant was great!', 'The curtains were disgusting']
    print(data)
    #data = ['The restaurant was great!', 'The curtains were disgusting']
    model_id = 'cl_pi3C7JiL'
    result = ml.classifiers.classify(model_id, data).body
    with open(fileName, 'w') as json_file:
        json.dump(result, json_file)
else:
    with open(fileName) as json_file:
        result = json.load(json_file)

pos=0
neg=0
neu=0
for item in result:
    r = item['classifications'][0]['tag_name']
    if r == "Positive":
        pos+=1
    if r == "Negative":
        neg+=1
    if r == "Neutral":
        neu+=1
print(f'{sys.argv[1]} pos={pos}, neg={neg}, neu={neu}')
