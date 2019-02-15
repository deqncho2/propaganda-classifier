from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
import json
import os
import pickle
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import tensorflow as tf
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import gc

tokenizer = TweetTokenizer()
stemmer = PorterStemmer()
stopWords = stopwords.words('english')
word_dict = {}
articles = []
tags = []

# For my sanity when running long jobs...
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

print("making article array")
if os.path.isfile("articles_and_tags.pkl"):
    pickleFile = open("articles_and_tags.pkl", 'rb')
    articles = pickle.load(pickleFile)
    if os.path.isfile("word_dict.json"):
        with open('word_dict.json') as f:
            word_dict = json.load(f)
else:
    with open('task1.train.txt', 'r') as file:
        l = 35993 # number of articles in dataset
        printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        unique_words = 0
        for idx,line in enumerate(file):
            words = line.split("\t")[0]
            tags.append(line.split("\t")[2].strip())
            article = []
            for j in tokenizer.tokenize(words):
                w = stemmer.stem(j.lower())
                if j.lower() in stopWords:
                    continue
                else:
                    if w not in word_dict.keys(): 
                        word_dict[w] = unique_words 
                        unique_words += 1
                    article.append(w)
            articles.append(article)
            printProgressBar(idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    with open('articles.pkl', 'wb') as f:
                   pickle.dump(articles, f)
    with open('tags.pkl', 'wb') as f:
                   pickle.dump(tags,f)
    json = json.dumps(word_dict)
    f = open("word_dict.json","w")
    f.write(json)
    f.close()


data = np.zeros((len(articles), len(word_dict.keys())))
print("making matrix")
l = len(articles)
# one hot encode data from article words dictionary into matrix form.
for idx,(article,tag) in enumerate(zip(articles,tags)):
    for word in article:
        data[idx][word_dict[word]] = 1
    printProgressBar(idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

columns = list(word_dict.keys())
df = pd.DataFrame(data, columns=columns)
# free memory for later or python is killed
del data
indices = np.random.permutation(len(articles))
train_data = df.iloc[indices[:5000]]
test_data = df.iloc[indices[len(indices)-5000:]]

# autoencoder used to reduce the dimensionality of the data by learning a latent space representation.
# Import that the model can be trained and can predict in batches so we don't have to load the 
# entire dataset into memory
class Autoencoder(object):

    def __init__(self, train, test, model_name="model", num_epochs=1000, learning_rate=0.4, batch_size=100, 
        denoising=False, masking=0, num_layers=1, num_hidden_1=256, num_hidden_2=128, num_hidden_3=64):
        
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.display_step = 1
        self.denoising = denoising
        self.masking = masking
        self.train = train 
        self.test = test
        
        tf.reset_default_graph()
        # Network Parameters
        self.num_hidden_1 = num_hidden_1 # 1st layer num features
        self.num_hidden_2 = num_hidden_2 # 2nd layer num features
        self.num_input = self.train.shape[1] 
        self.num_layers = num_layers

        # tf Graph input
        self.X = tf.placeholder("float", [None, None])
        self.Y = tf.placeholder("float", [None, None])
        tf.set_random_seed(1)
        # different shapes of weights and biases depending on how many layers we end up using.
        self.weights2 = {
            'encoder_h1': tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input])),
        }
        self.biases2 = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.num_input])),
        }
        self.weights1 = {
            'encoder_h1': tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input])),
        }
        self.biases1 = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([self.num_input])),
        }

        self.saver = tf.train.Saver()
        # choose size of model based on number of layers
        if num_layers==2:
            self.encoder_op = self.encoder2(self.X)
            self.decoder_op = self.decoder2(self.encoder_op)
        else:
            self.encoder_op = self.encoder1(self.X)
            self.decoder_op = self.decoder1(self.encoder_op)

        # Prediction
        self.y_pred = self.decoder_op
        # Targets (Labels) are the input data so that the model learns to reconstruct
        # the input data given the small number of dimensions in the hidden layers.
        self.y_true = self.Y

        # Define loss and optimizer, minimize the squared error
        self.loss = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        self.sess = tf.Session()
        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
    
    # Building the encoder
    def encoder1(self,x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights1['encoder_h1']),
                                       self.biases1['encoder_b1']))
        return layer_1

    # Building the decoder
    def decoder1(self,x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights1['decoder_h1']),
                                       self.biases1['decoder_b1']))
        return layer_1

    def encoder2(self,x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights2['encoder_h1']),
                                       self.biases2['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights2['encoder_h2']),
                                       self.biases2['encoder_b2']))
        return layer_2

    # Building the decoder
    def decoder2(self,x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights2['decoder_h1']),
                                       self.biases2['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights2['decoder_h2']),
                                       self.biases2['decoder_b2']))
        return layer_2

    def trainer(self, continue_from_saved=False, save=True):
        # Make sure to free memory in order to allow the model to keep training 
        # without python being killed.
        gc.collect()
        with tf.Session() as sess:

            # Run the initializer
            sess.run(tf.global_variables_initializer())
            if continue_from_saved:
                self.saver.restore(sess, self.model_name + '.ckpt')
            # Training
            train_loss = []
            test_loss = []
            for i in range(1, self.num_epochs):
                # Prepare Data
                batch_x = self.train.iloc[np.random.choice(self.train.shape[0], self.batch_size, replace=False)]
                batch_y = batch_x
                if self.denoising:
                    noise = np.random.normal(0.05, 0.1, batch_x.shape)
                    batch_x = np.add(batch_x,noise)
                if self.masking > 0:
                    pass
                batch_test = self.test.iloc[np.random.choice(self.test.shape[0], self.batch_size, replace=False)]

                # Run optimization op (backprop) and cost op (to get loss value)
                _, l = sess.run([self.optimizer, self.loss], feed_dict={self.X: batch_x, self.Y: batch_y})
                train_loss.append(l)
                _, loss_test = sess.run([self.decoder_op, self.loss], feed_dict={self.X: batch_test, self.Y: batch_test})
                test_loss.append(loss_test)
                # Display logs per step
                if i % self.display_step == 0 or i == 1:
                    print('Step %i: Minibatch Loss: %f  Test loss: %f' % (i, l, loss_test))
                # save the model every 20 epochs in case python is killed
                if save and i % 20 == 0:
                    self.saver.save(sess, "/Users/MaxGirkins/Documents/Work/uni_work/year_4/ttds/cw3/datathon/" + self.model_name + '.ckpt')
            sess.close()
            with open('autoencoder_loss.pkl', 'wb') as f:
                   pickle.dump(train_loss, f)

    # run the encoder operation on the data to reduce dimensionality
    def predict(self, matrix):
            res = self.sess.run([self.encoder_op], feed_dict={self.X: matrix, self.Y:matrix})
            return res

ae = Autoencoder(train_data, test_data, num_layers=2, num_epochs=1000, batch_size=250, model_name="propaganda_autoencoder")
#ae.trainer(continue_from_saved=True)
output = []
ae.saver.restore(ae.sess, ae.model_name + '.ckpt')
ae.sess.run(tf.global_variables_initializer())
l = len(df.values)/200
# save lower dimensional data to file in case of crashes after a batch.
for idx,i in enumerate([df.values[j:j + 200] for j in range(21600, len(df.values), 200)]):
    printProgressBar(idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    output.append(ae.predict(i))
    with open('minified_data/' + str(idx+108)+ 'minified_data.pkl', 'wb') as f:
        pickle.dump(output, f)
        output = []
ae.sess.close()






