# models.py


import numpy as np
import random
from collections import Counter
from sentiment_data import *
from utils import *



from collections import Counter
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """

        feats = Counter()
        for word in sentence:
            word = word.lower()  # normalize casing
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(f"UNI={word}")
            else:
                idx = self.indexer.index_of(f"UNI={word}")
                if idx == -1:
                    continue
            feats[idx] += 1  # count-based features

        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        for word in sentence:
            word = word.lower()
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(f"UNI={word}")
            else:
                idx = self.indexer.index_of(f"UNI={word}")
                if idx == -1:
                    continue
            feats[idx] = 1
        return feats



class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        for i in range(len(sentence) - 1):
            w1 = sentence[i].lower()
            w2 = sentence[i+1].lower()
            bigram = f"BIGRAM={w1}_{w2}"
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(bigram)
            else:
                idx = self.indexer.index_of(bigram)
                if idx == -1:
                    continue
            feats[idx] = 1  # Binary bigram
        return feats


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.vocab_counts = Counter()  # Used for filtering rare words

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        words = [w.lower() for w in sentence if w.lower() not in STOPWORDS]

        # Unigram features
        for word in words:
            if add_to_indexer or self.vocab_counts[word] > 1:
                feat_name = f"UNI={word}"
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(feat_name)
                else:
                    idx = self.indexer.index_of(feat_name)
                    if idx == -1:
                        continue
                feats[idx] = 1

        # Bigram features
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            bigram = f"BIGRAM={w1}_{w2}"
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(bigram)
            else:
                idx = self.indexer.index_of(bigram)
                if idx == -1:
                    continue
            feats[idx] = 1

        return feats



class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        feats = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = 0.0
        for feat_idx, value in feats.items():
            score += self.weights[feat_idx] * value
        return 1 if score >= 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):

    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        feats = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights[f] * v for f, v in feats.items())
        prob = 1 / (1 + np.exp(-score))  # Sigmoid
        return 1 if prob >= 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    
    num_epochs = 10
    indexer = feat_extractor.get_indexer()

    # First pass: build feature space
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    weights = np.zeros(len(indexer))

    for epoch in range(num_epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights[f] * v for f, v in feats.items())
            pred = 1 if score >= 0 else 0
            if pred != ex.label:
                for f, v in feats.items():
                    weights[f] += v if ex.label == 1 else -v

    return PerceptronClassifier(weights, feat_extractor)

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:

    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """

    num_epochs = 10
    learning_rate = 0.1
    indexer = feat_extractor.get_indexer()

    # Build feature space
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    weights = np.zeros(len(indexer))

    for epoch in range(num_epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)

            for f, v in feats.items():
                if f >= len(weights):
                    print(f"⚠️ Feature index {f} is out of bounds for weight vector of size {len(weights)}")

            score = sum(weights[f] * v for f, v in feats.items())
            pred = 1 / (1 + np.exp(-score))  # Sigmoid
            error = ex.label - pred
            for f, v in feats.items():
                weights[f] += learning_rate * error * v

    return LogisticRegressionClassifier(weights, feat_extractor)

def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        if isinstance(feat_extractor, BetterFeatureExtractor):
            for ex in train_exs:
                for word in ex.words:
                    feat_extractor.vocab_counts[word.lower()] += 1
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model