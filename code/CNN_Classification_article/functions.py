import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#Balances the data into equal parts propaganda and not.
def balance_data(texts, ys):

    balanced_texts = []
    balanced_ys = []
    counts = np.zeros(2)

    for i in range(len(texts)):
        y = ys[i]
        if counts[y] < 4000:
            balanced_texts.append(texts[i])
            balanced_ys.append(ys[i])
            counts[y] += 1

    return balanced_texts, balanced_ys

#First tokenises the data and then transforms each token into a number based the word index
def tokenize_and_transform(texts, num_words, maxlen):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=maxlen)

    word_index = tokenizer.word_index

    return data, word_index

#Creates the embedding matrix
def embed_matrix(embedding, word_index, dim):
    if embedding == "glove":
        return load_glove(word_index, dim)
    else:
        return load_w2v(word_index, dim)

#Loads the pretrained glove embedding file
def load_glove(word_index, dim):

    print("Generating Glove Embedding")

    matrix_gl = np.zeros((max(word_index.values()) + 1, dim), dtype = np.float32)
    seen = set()
    f = open('data/embeddings/glove/glove.twitter.27B.%dd.txt' % dim)
    for line in f:
        values = line.strip().split(' ')
        if len(values) == dim+1:
            word = values[0]
            if word in word_index:
                matrix_gl[word_index[word]] = [float(x) for x in values[1:]]
                seen.add(word)
                if len(seen) == len(word_index):
                    break
    f.close()
    return matrix_gl

#Loads the pretrained w2v embedding file
def load_w2v(word_index, dim):
    from gensim.models import word2vec

    print("Generating W2V Embedding")

    path = "data/embeddings/w2v/text8_%dd_model" % dim
    wvmodel = word2vec.Word2Vec.load(path)

    matrix_wv = np.zeros((max(word_index.values()) + 1, dim), dtype = np.float32)
    seen = set()
    for word in wvmodel.wv.vocab.keys():
        if word in word_index:
            matrix_wv[word_index[word]] = wvmodel.wv[word]
            seen.add(word)
            if len(seen) == len(word_index):
                break
    return matrix_wv
