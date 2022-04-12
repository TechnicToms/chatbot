import numpy as np
import tensorflow as tf
import tweepy
import tqdm


class TerminalColor:
    info = "\u001b[1m\u001b[34m[INFO] \u001b[0m"
    success = "\u001b[1m\u001b[32;1m[SUCCESS] \u001b[0m"
    error = "\u001b[1m\u001b[31m[ERROR] \u001b[0m"
    warning = "\u001b[1m\u001b[33;1m[WARNING] \u001b[0m"


class data_loader:
    def __init__(self, path2dataset):
        """
        Loads the Dataset

        :param path2dataset: Path to .json file
        :type path2dataset: str
        """
        self.tc = TerminalColor
        self.setup_twitter_api()
        self.load_tweets(path2dataset)

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

    def load_tweets(self, path2dataset):
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
        for ids in tqdm.tqdm(twitter_ids, desc="Loading Tweets"):
            # Only pull tweets with full conversation
            try:
                tweets = [self.api.get_status(ids[0]), self.api.get_status(ids[1]), self.api.get_status(ids[2])]
                with open("tweets.txt", "a") as f:
                    for tweet in tweets:
                        print(tweet.text.encode('UTF-8'))
                        f.write(tweet.text.encode('UTF-8'))
                        f.write(" | ")
                    f.write("\n")
            except tweepy.TweepyException:
                missing_tweets += 1

        if missing_tweets > 0:
            print(self.tc.warning + f"{missing_tweets} tweets could not be loaded because they no longer exist!")
        print(self.tc.success + "Loaded Data!")


if __name__ == '__main__':
    data = data_loader("datasets/MSRSocialMediaConversationCorpus/twitter_ids.tuning.txt")
    print("finished!")
