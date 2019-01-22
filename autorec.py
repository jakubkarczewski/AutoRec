"""This is heavily inspired by AutoRec paper,
 http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf.
 """

import argparse
from os.path import isfile, isdir, join
from os import mkdir
import pickle

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE


class AutoRec:
    def __init__(self, ratings_path, movies_path, batch_size, epochs, hl_size, learning_rate,
                 dataset_split_ratio=0.8, save_dir='./models/'):
        for file_path in (ratings_path, movies_path):
            assert isfile(file_path), "{} is not valid file path".format(file_path)

        self.ratings_path = ratings_path
        self.movies_path = movies_path

        self.ratings = None
        self.movies = None
        self.num_movies = None
        self.num_users = None
        self.model_dict = None

        self.hl_size = hl_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dataset_split_ratio = dataset_split_ratio

        if not isdir(save_dir):
            mkdir(save_dir)
        self.save_dir = save_dir

    def _load_dataframes(self):
        """Loads .dat files as Pandas Dataframes."""
        self.ratings = pd.read_csv(self.ratings_path, sep="::", header=None, engine='python')
        self.movies = pd.read_csv(self.movies_path, sep="::", header=None, engine='python')
        # get number of columns (movies) from user-item matrix
        self.num_movies = self.ratings.shape[-1]

    def prepare(self):
        """Runs necessary preprocessing and building."""
        print('Preparing data and model ...')
        self._load_dataframes()
        self._build_model()

    def _build_model(self):
        """Defines model as computation graph."""
        # define hidden layer weights
        hidden_weights = {'weights': tf.Variable(tf.random_normal([self.num_movies + 1, self.hl_size]))}
        # define output layer weights
        output_weights = {'weights': tf.Variable(tf.random_normal([self.hl_size + 1, self.num_movies]))}

        # define input layer along with bias
        input_layer = tf.placeholder('float', [None, self.num_movies])
        input_layer_bias = tf.fill([tf.shape(input_layer)[0], 1], 1.0)
        input_layer_concat = tf.concat([input_layer, input_layer_bias], 1)

        # define hidden layer along with activation function and bias
        hidden_layer = tf.nn.sigmoid(tf.matmul(input_layer_concat, hidden_weights['weights']))
        hidden_layer_bias = tf.fill([tf.shape(hidden_layer)[0], 1], 1.0)
        hidden_layer_concat = tf.concat([hidden_layer, hidden_layer_bias], 1)

        # define output layer
        output_layer = tf.matmul(hidden_layer_concat, output_weights['weights'])

        # define ground truth layer
        ground_truth_layer = tf.placeholder('float', [None, self.num_movies])

        # define loss function (MSE)
        loss = tf.reduce_mean(tf.square(output_layer - ground_truth_layer))

        # define optimizer
        optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)

        self.model_dict = {'optimizer': optimizer, 'loss': loss, 'input': input_layer, 'gt': ground_truth_layer,
                           'output': output_layer}

        return self.model_dict

    def _create_datasets(self):
        """Creates and returns datasets from raw Dataframes."""
        x_train, x_test = train_test_split(self.ratings, train_size=self.dataset_split_ratio)
        return x_train, x_test

    def train(self):
        """Run training of the network."""
        if not self.ratings:
            self.prepare()

        # initialize variables and start the session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # initialize saver
        saver = tf.train.Saver()

        # get training data
        x_train, x_test = self._create_datasets()

        # serialize testing data for later inference
        with open('./test.pkl', 'wb') as f:
            pickle.dump(x_test, f)

        self.num_users = int(x_train.shape[0])

        # run training loop
        for epoch in range(self.epochs):
            temp_loss = 0
            for index in range(self.num_users//self.batch_size):
                epoch_data = x_train[index*self.batch_size:(index+1)*self.batch_size]
                _, partial_loss = sess.run([self.model_dict['optimizer'], self.model_dict['loss']],
                                           feed_dict={self.model_dict['input']: epoch_data,
                                                      self.model_dict['gt']: epoch_data})
                temp_loss += partial_loss

                output_train = sess.run(self.model_dict['output'], feed_dict={self.model_dict['input']: x_train})
                output_test = sess.run(self.model_dict['output'], feed_dict={self.model_dict['input']: x_test})

            if epoch % 50 == 0:
                save_path = saver.save(sess, join(self.save_dir, 'model_e:{}_l:{}.ckpt'.format(epoch, temp_loss)))
                print("Model saved in path: %s" % save_path)

            # log progress
            print('Epoch {}/{}, loss: {}'.format(epoch, self.epochs, temp_loss))
            print('Training MSE:', MSE(output_train, x_train))
            print('Testing MSE:', MSE(output_test, x_test))

    def infer(self, model_path, user_id):
        """Return predicted ratings for user from test set."""
        if not self.ratings:
            self.prepare()

        # reset graph
        tf.reset_default_graph()

        # initialize saver
        saver = tf.train.Saver()

        # initialize
        sess = tf.Session()
        saver.restore(sess, model_path)

        with open('./test.pkl', 'rb') as f:
            x_test = pickle.load(f)

        try:
            user_data = x_test.iloc[user_id, :]
        except Exception as ex:
            print("Wrong user id, try different one.\n", str(ex))
            exit()

        user_pred = sess.run(self.model_dict['output'],
                             feed_dict={self.model_dict['input']: [user_data]})
        return user_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        help="Either train or infer")
    parser.add_argument("--ratings_path", type=str, required=True,
                        help="Path to ratings data.")
    parser.add_argument("--movies_path", type=str, required=True,
                        help="Path to movies data.")
    parser.add_argument("--model_path", type=str, required=False,
                        help="Path to movies data.")
    parser.add_argument("--user_id", type=str, required=False,
                        help="Id for inference.")
    args = parser.parse_args()

    ar = AutoRec(args.ratings_path, args.movies_path, 100, 200, 256, 0.1)
    if args.mode == 'train':
        ar.train()
    elif args.mode == 'infer':
        assert isfile(args.model_path)
        ar.infer(args.model_path, args.user_id)

