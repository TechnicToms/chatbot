import numpy as np
import tweepy
import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from sklearn.model_selection import train_test_split

import unicodedata
import re

import os
import io
import time


class TerminalColor:
    info = "\u001b[1m\u001b[34m[INFO] \u001b[0m"
    success = "\u001b[1m\u001b[32;1m[SUCCESS] \u001b[0m"
    error = "\u001b[1m\u001b[31m[ERROR] \u001b[0m"
    warning = "\u001b[1m\u001b[33;1m[WARNING] \u001b[0m"


class data_loader:
    num_encoder_tokens = 0
    num_decoder_tokens = 0

    def __init__(self, dataset):
        """
        Loads the Dataset

        :param dataset: Selected Dataset. Possible answers ['chat-corpus', 'kaggle-1']
        :type dataset: str
        """
        self.dataset = []
        self.target_tokens = []
        self.input_tokens = []

        self.tc = TerminalColor

        if dataset == "chat-corpus":
            self.load_tweets_textfile("datasets/Chat corpus/twitter_en_big.txt")
        if dataset == "kaggle-1":
            with open("datasets/kaggle/chatbot-1/dialogs.txt", "r") as fileReader:
                lines = fileReader.readlines()
                self.dataset = [line.split('-|-') for line in lines]
            print(self.tc.success + f"Found {len(self.dataset)} messages pairs!")

            self.tokenize()
        else:
            print(self.tc.error + f"The given Dataset {dataset} didn't exists!")

    def __len__(self):
        return len(self.dataset)

    def setup_twitter_api(self, path_to_file="secret_keys.txt"):
        """
        Setups the api for Twitter

        :param path_to_file: Secret File
        :type path_to_file: str
        :return: None
        """
        # Read my secret key
        # Keys = [API Key, API Key Secret, Bearer Token, Access Token, Access Token Secret, Client ID, Client Secret]
        with open(path_to_file) as fp:
            keys = fp.readlines()

        APP_KEY = keys[0].replace("\n", "")
        APP_SECRET = keys[1].replace("\n", "")
        BEARER_TOKEN = keys[2].replace("\n", "")
        ACCESS_TOKEN = keys[3].replace("\n", "")
        ACCESS_TOKEN_SECRET = keys[4].replace("\n", "")

        auth = tweepy.OAuth1UserHandler(APP_KEY, APP_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

        self.api = tweepy.API(auth)

    def load_tweets_by_ids(self, path2dataset):
        """
        Loads the Tweets into an text file

        :param path2dataset: Path to Files
        :type path2dataset: str
        :return: None
        """
        # Load File
        twitter_ids = np.genfromtxt(path2dataset, delimiter=",")

        # Reset Counter
        missing_tweets = 0

        # Loop over IDs
        for id in tqdm.tqdm(twitter_ids, desc="Loading Tweets"):
            # Only pull tweets with full conversation
            try:
                tweet = self.api.get_status(id)
                print(tweet)
                with open("tweets.txt", "a") as f:
                    print(tweet.text.encode('UTF-8'))
                    f.write(tweet.text.encode('UTF-8'))
                    f.write(",")
                    f.write("\n")
            except tweepy.TweepyException:
                missing_tweets += 1

        if missing_tweets > 0:
            print(self.tc.warning + f"{missing_tweets} tweets could not be loaded because they no longer exist!")
        print(self.tc.success + "Loaded Data!")

    def load_tweets_textfile(self, textfile_path):
        """
        Loads the Messages from a text file

        :param textfile_path: Path to file
        :type textfile_path: str
        :return: Sets Dataset List
        """
        with open(textfile_path, "r") as fileReader:
            self.dataset = fileReader.readlines()
        print(self.tc.success + f"Found {len(self.dataset)} Messages!")

    def tokenize(self):
        """
        Tokenizes the Dataset!

        :return: None
        """
        input_docs = []
        target_docs = []
        input_tokens = set()
        target_tokens = set()

        for line in self.dataset:
            input_doc, target_doc = line[0], line[1]
            # Appending each input sentence to input_docs
            input_docs.append(input_doc)
            # Splitting words from punctuation
            target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
            # Redefine target_doc below and append it to target_docs
            target_doc = '<START> ' + target_doc + ' <END>'
            target_docs.append(target_doc)

            # Now we split up each sentence into words and add each unique word to our vocabulary set
            for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
                if token not in input_tokens:
                    input_tokens.add(token)
            for token in target_doc.split():
                if token not in target_tokens:
                    target_tokens.add(token)
        input_tokens = sorted(list(input_tokens))
        target_tokens = sorted(list(target_tokens))
        self.num_encoder_tokens = len(input_tokens)
        self.num_decoder_tokens = len(target_tokens)

        print(self.tc.info + f"#encoder-tokens: {self.num_encoder_tokens} | "
                             f"#decoder-tokens: {self.num_decoder_tokens}")

        input_features_dict = dict(
            [(token, i) for i, token in enumerate(input_tokens)])
        target_features_dict = dict(
            [(token, i) for i, token in enumerate(target_tokens)])
        reverse_input_features_dict = dict(
            (i, token) for token, i in input_features_dict.items())
        reverse_target_features_dict = dict(
            (i, token) for token, i in target_features_dict.items())

        # Maximum length of sentences in input and target documents
        max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
        max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])
        encoder_input_data = np.zeros(
            (len(input_docs), max_encoder_seq_length, self.num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(input_docs), max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(input_docs), max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')

        for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
            for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
                # Assign 1. for the current line, timestep, & word in encoder_input_data
                encoder_input_data[line, timestep, input_features_dict[token]] = 1.

            for timestep, token in enumerate(target_doc.split()):
                decoder_input_data[line, timestep, target_features_dict[token]] = 1.
                if timestep > 0:
                    decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.

        print(self.tc.success + "Tokenized!")

    def get_sample(self):
        choosen = np.random.randint(0, len(self))
        return self.input_tokens[choosen]


class Encoder(tf.keras.Model):
    def __init__(self, num_encoder_tokens, dimensionality):
        super().__init__(self)
        # Build Encoder
        encoder_inputs = tf.keras.layers.Input(shape=(None, num_encoder_tokens))
        encoder_lstm = tf.keras.layers.LSTM(dimensionality, return_state=True)
        encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
        encoder_states = [state_hidden, state_cell]


if __name__ == '__main__':
    data = data_loader("kaggle-1")
    print(data.get_sample())

    BUFFER_SIZE = 32000
    BATCH_SIZE = 64
    EPOCHS = 10
    # Let's limit the #training examples for faster training
    num_examples = 30000
    # tu = training_unit(BATCH_SIZE, EPOCHS, BUFFER_SIZE, num_examples)

    print("finished!")
