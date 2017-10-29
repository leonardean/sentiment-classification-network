import time
import sys
import numpy as np
from collections import Counter

class SentimentNetwork:
    def __init__(self, reviews, labels, min_count, polarity_cutoff, hidden_nodes = 10, learning_rate = 0.1):
        np.random.seed(1)
        self.pre_process_data(reviews, labels, min_count, polarity_cutoff)
        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels, min_count, polarity_cutoff):

        # initialize counters and vocab set
        total_counts = Counter()
        positive_counts = Counter()
        negative_counts = Counter()
        pos_neg_ratios = Counter()
        review_vocab = set()
        label_vocab = set()

        # calculate positive-negative ratios for all distinct words
        for review, label in zip(reviews, labels):
            for word in review.split(' '):
                total_counts[word] += 1
                if label == 'POSITIVE':
                    positive_counts[word] += 1
                elif label == 'NEGATIVE':
                    negative_counts[word] += 1

        for word in total_counts.keys():
            pos_neg_ratios[word] = positive_counts[word] / float(negative_counts[word] + 1)
            if pos_neg_ratios[word] > 1:
                pos_neg_ratios[word] = np.log(pos_neg_ratios[word])
            else:
                pos_neg_ratios[word] = -np.log(1 / (pos_neg_ratios[word] + 0.01))

        # calculate vocab for reviews and labels
        for review, label in zip(reviews, labels):
            label_vocab.add(label)
            for word in review.split(' '):
                # we only add words after filtering to the vocab, so that we can tune the filters later
                if total_counts[word] >= min_count and np.abs(pos_neg_ratios[word]) >= polarity_cutoff:
                    review_vocab.add(word)
        self.review_vocab = list(review_vocab)
        self.label_vocab = list(label_vocab)
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # turn words to indices to facilitate network training
        self.word2index = {}
        for index, word in enumerate(self.review_vocab):
            self.word2index[word] = index

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # initialize weights between layers
        self.weights_0_1 = np.random.normal(0.0, self.input_nodes**-0.5,
                                           (self.input_nodes, self.hidden_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                           (self.hidden_nodes, self.output_nodes))

        # initialize hidden layer with zeros
        # we jump input layer because we will do matrix calculation for input layer with our own way to
        # reduce unneccesary calculations (skip the zero multiplications and additions)
        self.layer_1 = np.zeros((1, hidden_nodes))

    def get_target_for_label(self, label):
        if label == 'POSITIVE':
            return 1
        elif label == 'NEGATIVE':
            return 0

    # activation function of output layer
    def sigmoid(self, x):
        return 1 / (1 + np.exp( -x ))

    # derivative of activation function
    def sigmoid_prime(self, output):
        return output * (1 - output)

    def train(self, training_reviews, training_labels):
        assert(len(training_reviews) == len(training_labels))

        # keep track of correct predictions for accuracy display
        correct_so_far = 0
        # we also want to track the training efficiency of the network
        start = time.time()

        for i in range(len(training_reviews)):

            review = training_reviews[i]
            label = training_labels[i]
            # reset hidden layer value
            self.layer_1 *= 0

            # calculate the word indices for current review
            review_indice = set()
            for word in review.split(' '):
                if word in self.word2index.keys():
                    review_indice.add(self.word2index[word])
            review_indice_list = list(review_indice)

            # hidden layer calculation. by doing so, we drastically reduce the complexity of matrix calculation
            for index in review_indice_list:
                self.layer_1 += self.weights_0_1[index]
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

            # backward propagate and weights update
            layer_2_error = layer_2 - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * self.sigmoid_prime(layer_2)

            layer_1_error = np.dot(layer_2_delta, self.weights_1_2.T)
            layer_1_delta = layer_1_error

            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate
            for index in review_indice_list:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate

            # print training progress info
            if (np.abs(layer_2_error) < 0.5):
                correct_so_far += 1
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")

            sys.stdout.flush()

    def test(self, testing_reviews, testing_labels):
        correct = 0
        start = time.time()

        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if pred == testing_labels[i]:
                correct += 1

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
            sys.stdout.flush()

    def run(self, review):
        self.layer_1 *= 0
        review_indice = set()

        for word in review.split(' '):
            if word in self.word2index.keys():
                review_indice.add(self.word2index[word])

        review_indice_list = list(review_indice)

        for index in review_indice_list:
            self.layer_1 += self.weights_0_1[index]
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

        if layer_2 >= 0.5:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'
