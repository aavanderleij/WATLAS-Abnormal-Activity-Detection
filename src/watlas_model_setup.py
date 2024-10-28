"""
setup for watlas disturbance model
"""
import sys

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class ModelSetup:

    def __init__(self):
        # TODO add  exact species per group
        # TODO add time of day/time to high tide when more training data is available
        # TODO add tidal data
        # columns by type
        self.columns_to_drop = ['TAG', 'time', 'X', 'Y', 'TIME', 'tag', 'X_raw', 'Y_raw']
        self.numerical_columns = ['NBS', 'VARX', 'VARY', 'distance', 'speed_in', 'speed_out', 'turn_angle',
                                  'mean_dist_group', 'median_dist_group', 'std_dist_group', 'group_size',
                                  'n_species_group', 'mean_turn_angle_group', 'median_turn_angle_group',
                                  'std_turn_angle_group', 'mean_speed_group', 'speed_median_group', 'speed_std_group',
                                  'speed_in_1', 'speed_out_1', 'distance_1', 'turn_angle_1', 'group_size_1',
                                  'mean_turn_angle_group_1', 'mean_dist_group_1', 'mean_speed_group_1', 'speed_in_2',
                                  'speed_out_2', 'distance_2', 'turn_angle_2', 'group_size_2',
                                  'mean_turn_angle_group_2', 'mean_dist_group_2', 'mean_speed_group_2', 'speed_in_3',
                                  'speed_out_3', 'distance_3', 'turn_angle_3', 'group_size_3',
                                  'mean_turn_angle_group_3', 'mean_dist_group_3', 'mean_speed_group_3', 'speed_in_4',
                                  'speed_out_4', 'distance_4', 'turn_angle_4', 'group_size_4',
                                  'mean_turn_angle_group_4', 'mean_dist_group_4', 'mean_speed_group_4', 'waterlevel']
        self.categorical_sting_columns = ["species"]
        self.categorical_bool_columns = ['islandica_in_group', 'oystercatcher_in_group', 'spoonbill_in_group',
                                         'bar-tailed_godwit_in_group', 'redshank_in_group', 'sanderling_in_group',
                                         'dunlin_in_group', 'turnstone_in_group', 'curlew_in_group',
                                         "gray_plover_in_group"]

        self.metrics = [
            keras.metrics.BinaryCrossentropy(name='cross entropy'),
            keras.metrics.MeanSquaredError(name='Brier score'),
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'),
        ]

    def load_train_data(self, path_to_traindata="watlas_with_labels.csv"):
        """
        Read training data from csv file and load it into a pandas dataframe.
        Args:
            path_to_traindata (str): path to training data csv file:

        Returns:
            train_data (pandas dataframe): training data dataframe

        """
        # read csv file
        # TODO check columns
        train_data_df = pd.read_csv(path_to_traindata)
        # TODO keep an eye on parameters
        # drop columns not used for training
        train_data_df = train_data_df.drop(columns=self.columns_to_drop)

        # check for nans and inf
        # print(train_data_df.isna().sum())
        # print(train_data_df.isin([np.inf, -np.inf]).sum())

        print(train_data_df.columns)
        return train_data_df

    def split_data(self, dataframe):
        """
        split training data into train, validation and test sets
        Args:
            dataframe (pandas dataframe): training data dataframe

        Returns:
            train (pandas dataframe): training data dataframe
            val (pandas dataframe): validation data dataframe
            test (pandas dataframe): test data dataframe

        """
        neg, pos = np.bincount(dataframe['Alert'])
        total = neg + pos
        print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
            total, pos, 100 * pos / total))

        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)

        class_weight = {0: weight_for_0, 1: weight_for_1}

        # set initial bais
        initial_bias = np.log([pos / neg])
        self.output_bias = keras.initializers.Constant(initial_bias)

        print('Weight for class 0: {:.2f}'.format(weight_for_0))
        print('Weight for class 1: {:.2f}'.format(weight_for_1))

        # split training data into training data set, validation data set and a test data set
        # TODO use scikit train test split to get random proportinal validation sets
        train, test = train_test_split(dataframe, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2)

        neg, pos = np.bincount(test['Alert'])
        total = neg + pos
        print('Examples test:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
            total, pos, 100 * pos / total))

        print(fr"Amount of data {total} ")

        # print amount of samples in each set
        print(len(train), 'training examples')
        print(len(val), 'validation examples')
        print(len(test), 'test examples')
        return train, val, test, class_weight

    def df_to_dataset(self, dataframe, shuffle=False, batch_size=32):
        """
        converts dataframe to TensorFlow dataset
        boilerplate code from TensorFlow
        Args:
            dataframe (pandas dataframe): input dataframe
            shuffle (bool): whether to shuffle the dataframe. Defaults to True.
            batch_size (int): size of mini-batches. Defaults to 32.

        Returns:
            ds: tf.data.Dataset
        """
        # make a copy if the dataframe
        df = dataframe.copy()
        # separate labels
        labels = df.pop("Alert")
        # make a dict with column name as key, value is numpy array and reshape the vector to an 1D array (n, 1)
        df = {key: value.to_numpy()[:, tf.newaxis] for key, value in dataframe.items()}
        # convert to dataset object
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        # if shuffle true, shuffle
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        # groups dataset into batches
        ds = ds.batch(batch_size)
        # prefetch dataset for optimization
        ds = ds.prefetch(batch_size)

        return ds

    def get_normalization_layer(self, name, dataset):
        """
        Computes the mean and variance of the feature specified by name, in a dataset.
        Returns a normalization layer for the feature.

        Args:
            name (str): The name of the numeric feature/column in the dataset to be normalized.
            dataset (tf.data.Dataset): TensorFlow dataset with numeric features.

        Returns:
            tf.keras.layers.Normalization: A normalization layer that standardizes the feature
                                       based on the dataset statistics (mean and variance).
        """
        # Create a Normalization layer for the feature.
        normalizer = layers.Normalization(axis=None)

        # Prepare a Dataset that only returns the feature specified by name.
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)

        # return configured normalization layer for the feature.
        return normalizer

    def get_category_encoding_layer(self, name, dataset, dtype, max_tokens=None):
        """
        Encodes the categorical features to one-hot encoding.

        Args:
            name (str): The name of the feature/column in the dataset to be encoded.
            dataset (tf.data.Dataset): A TensorFlow dataset with features to be encoded.
            dtype (str): The data type of the feature. Expected values are 'string' for text features
                     or other types for integer features.
            max_tokens (int, None): The maximum number of unique tokens (categories), if None . Default is None.

        Returns:
            Encoded feature

        """

        if dtype == 'string':
            # Create a layer that turns strings into integer indices.
            index = layers.StringLookup(max_tokens=max_tokens)
        else:
            # create a layer that turns integer values into integer indices.
            index = layers.IntegerLookup(max_tokens=max_tokens)

        # Prepare a `tf.data.Dataset` that only yields the feature.
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the set of possible values and assign them a fixed integer index.
        index.adapt(feature_ds)

        # Encode the integer indices.
        encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

        # Apply multi-hot encoding to the indices. The lambda function captures the
        # layer, so you can use them, or include them in the Keras Functional model later.
        return lambda feature: encoder(index(feature))

    def encode_features(self, dataset):
        """
        Encode features based on type of value (catagorical or numarical).

        categorical values wil be encoded using one-hot encoding.

        Args:
            dataset (tf.data.Dataset): A TensorFlow dataset with features to be encoded.

        Returns:

        """
        all_inputs = {}
        encoded_features = []

        # Numerical features.
        for header in self.numerical_columns:
            # create an input tensor for every header in numerical_columns
            numeric_col = tf.keras.Input(shape=(1,), name=header)
            # get normalization layer for header
            normalization_layer = self.get_normalization_layer(header, dataset)
            # Apply normalization
            encoded_numeric_col = normalization_layer(numeric_col)
            # Add columns input layer
            all_inputs[header] = numeric_col
            encoded_features.append(encoded_numeric_col)

        for header in self.categorical_sting_columns:
            # create an input tensor for every header in categorical_columns
            categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
            # get normalization layer for header
            encoding_layer = self.get_category_encoding_layer(name=header,
                                                              dataset=dataset,
                                                              dtype='string',
                                                              max_tokens=9)
            # Apply normalization
            encoded_categorical_col = encoding_layer(categorical_col)
            # Add columns input layer
            all_inputs[header] = categorical_col
            encoded_features.append(encoded_categorical_col)

        for header in self.categorical_bool_columns:
            bool_col = tf.keras.Input(shape=(1,), name=header)
            all_inputs[header] = bool_col
            encoded_features.append(bool_col)

        return all_inputs, encoded_features

    def create_model(self, encoded_features, all_inputs):
        """
        creates a compiled model

        Args:
            encoded_features (list): A list with encoded features
            all_inputs (dict): A dictionary of input tensors for eacht feature, key is feature name, value is
             corresponding tensor.

        Returns:

        """
        # Concat all features into single tensor
        all_features = tf.keras.layers.concatenate(encoded_features)
        # dense layer
        x = tf.keras.layers.Dense(64, activation="relu")(all_features)
        # dropout layer
        x = tf.keras.layers.Dropout(0.5)(x)
        # dense layer
        x = tf.keras.layers.Dense(32, activation="relu")(all_features)
        # dropout layer
        x = tf.keras.layers.Dropout(0.5)(x)
        # output layer
        output = tf.keras.layers.Dense(1, activation="sigmoid", bias_initializer=self.output_bias)(x)

        # create model
        model = tf.keras.Model(all_inputs, output)

        # compile model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=self.metrics,
                      run_eagerly=True)
        return model

    def create_model_zero_Bias(self, encoded_features, all_inputs):
        """
        creates a compiled model

        Args:
            encoded_features (list): A list with encoded features
            all_inputs (dict): A dictionary of input tensors for eacht feature, key is feature name, value is
             corresponding tensor.

        Returns:

        """
        # Concat all features into single tensor
        all_features = tf.keras.layers.concatenate(encoded_features)
        # dense layer
        x = tf.keras.layers.Dense(64, activation="relu")(all_features)
        # dropout layer
        x = tf.keras.layers.Dropout(0.5)(x)
        # dense layer
        x = tf.keras.layers.Dense(32, activation="relu")(all_features)
        # dropout layer
        x = tf.keras.layers.Dropout(0.5)(x)
        # output layer
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        # create model
        model = tf.keras.Model(all_inputs, output)

        # compile model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=self.metrics,
                      run_eagerly=True)
        return model


    def plot_metrics(self, history):
        metrics = ['loss', 'prc', 'precision', 'recall']
        for n, metric in enumerate(metrics):
            name = metric.replace("_", " ").capitalize()
            plt.subplot(2, 2, n + 1)
            plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_' + metric],
                     color=colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8, 1])
            else:
                plt.ylim([0, 1])

            plt.legend()
        plt.show()

    def plot_loss(self, history, label, n):
        # Use a log scale on y-axis to show the wide range of values.
        plt.semilogy(history.epoch, history.history['loss'],
                     color=colors[n], label='Train ' + label)
        plt.semilogy(history.epoch, history.history['val_loss'],
                     color=colors[n], label='Val ' + label,
                     linestyle="--")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def train_model(self):
        print("loading data...")
        train_data = self.load_train_data()
        print("split data...")
        train, val, test, class_weight = self.split_data(train_data)
        train_set = self.df_to_dataset(train, batch_size=25)
        val_set = self.df_to_dataset(train, batch_size=25)
        test_set = self.df_to_dataset(train, batch_size=25)
        [(train_features, label_batch)] = train_set.take(1)

        print('Every feature:', list(train_features.keys()))

        print("encode features")
        all_inputs, encoded_features = self.encode_features(train_set)
        print("compile model")
        ml_model = self.create_model(encoded_features, all_inputs)
        zero_model = self.create_model_zero_Bias(encoded_features, all_inputs)

        print("fit model")
        history = ml_model.fit(train_set,
                               epochs=10,
                               validation_data=val_set,
                               class_weight=class_weight)

        zero_bias_history = zero_model.fit(train_set,
                                epochs=10,
                                validation_data=val_set,
                                class_weight=class_weight)

        self.plot_metrics(history)
        self.plot_metrics(zero_bias_history)

        self.plot_loss(history, "carefull bias", 0)
        self.plot_loss(zero_bias_history, "zero bias", 1)

        print(ml_model.summary())

        ml_model.save("Wrong_model", save_format="tf")
        print("get result")
        result = ml_model.evaluate(test_set, return_dict=True)

        return result

    def test_tensorflow_install(self):
        """"
        Test if tensorflow is installed correctly
        """
        print(tf.reduce_sum(tf.random.normal([1000, 1000])))


def main():
    ms = ModelSetup()

    result = ms.train_model()
    # print(result)

    loaded_model = tf.keras.models.load_model("Wrong_model")

    train_data_df = pd.read_csv("watlas_all.csv")
    # TODO keep an eye on parameters
    # drop columns not used for training
    train_data_df = train_data_df.drop(columns=ms.columns_to_drop)

    df = {key: value.to_numpy()[:, tf.newaxis] for key, value in train_data_df.items()}

    results = loaded_model.predict(df)

    print(train_data_df.shape[0])
    print(len(results))
    print("probabilities:")
    print(results)
    train_data_df['probability'] = results
    train_data_df['prediction'] = tf.cast(results >= 0.5, tf.int32).numpy()

    train_data_df.to_csv("predictions_example.csv")


if __name__ == "__main__":
    sys.exit(main())
