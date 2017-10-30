import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
from collections import Counter

reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)

total_counts = Counter()
for _, row in reviews.iterrows():
    for word in row[0].split(' '):
        total_counts[word] += 1

vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]
word2index = {}
for index, word in enumerate(vocab):
    word2index[word] = index

def text_to_vector(text):
    word_vector = np.zeros(len(vocab))
    for word in text.split(' '):
        if word in vocab:
            word_vector[word2index[word]] = 1
    return word_vector

word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
    word_vectors[ii] = text_to_vector(text[0])

Y = (labels == 'positive').astype(np.int_)
records = len(labels)

shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records * test_fraction)], shuffle[int(records * test_fraction):]
trainX, trainY = word_vectors[train_split, :], to_categorical(Y.values[train_split].T[0], 2)
testX, testY = word_vectors[test_split, :], to_categorical(Y.values[test_split].T[0], 2)

def build_model():
    net = tflearn.input_data([None, 10000])
    net = tflearn.fully_connected(net, 200, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')

    model = tflearn.DNN(net)
    return model

model = build_model()
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=30)

predictions = (np.array(model.predict(testX))[:, 0] >= 0.5).astype(np.int_)
test_accuracy = np.mean(predictions == testY[:, 0], axis=0)
print("Test accuracy: ", test_accuracy)
