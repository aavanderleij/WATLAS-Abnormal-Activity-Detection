"""
setup for watlas disturbance model
"""
import sys

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers


class ModelSetup:

    def __init__(self):
        # TODO add  exact species per group
        # TODO add time of day/time to high tide when more training data is available
        # TODO add tidal data
        # columns by type
        self.columns_to_drop = ["df_id", "time_agg", "tag", "posID", "TAG", "large_disturb", "small_disturb", "time",
                                "TIME", "large_disturb_1", "large_disturb_2", "large_disturb_3", "large_disturb_4",
                                'small_disturb_1', 'small_disturb_2', 'small_disturb_3', 'small_disturb_4', "speed_out", 'alert_1', 'alert_2', 'alert_3', 'alert_4']
        self.numerical_columns = ['NBS', 'speed_in', 'VARX', 'VARY', 'group_size_per_timestamp',
                                  'n_unique_species', 'avg_dist', 'std_dist', 'distance', 'speed', 'turn_angle',
                                  'speed_1', 'distance_1', 'turn_angle_1', 'speed_2', 'distance_2', 'turn_angle_2',
                                  'speed_3', 'distance_3', 'turn_angle_3', 'speed_4', 'distance_4', 'turn_angle_4']
        self.categorical_sting_columns = ["species"]
        # self.categorical_bool_columns = ["alert", 'alert_1', 'alert_2', 'alert_3', 'alert_4']

    def load_train_data(self, path_to_traindata="data/training_data/processed_train_data.csv"):
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
        print(train_data_df.isna().sum())
        print(train_data_df.isin([np.inf, -np.inf]).sum())
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
        # split training data into training data set, validation data set and a test data set
        train, val, test = np.split(dataframe.sample(frac=1), [int(0.8 * len(dataframe)), int(0.9 * len(dataframe))])
        # print amount of samples in each set
        print(len(train), 'training examples')
        print(len(val), 'validation examples')
        print(len(test), 'test examples')
        return train, val, test

    def df_to_dataset(self, dataframe, shuffle=True, batch_size=32):
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
        labels = df.pop("alert")
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

        # for header in self.categorical_bool_columns:
        #     bool_col = tf.keras.Input(shape=(1,), name=header)
        #     all_inputs[header] = bool_col
        #     encoded_features.append(bool_col)

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
        x = tf.keras.layers.Dense(32, activation="relu")(all_features)
        # dropout layer
        x = tf.keras.layers.Dropout(0.5)(x)
        # output layer
        output = tf.keras.layers.Dense(1)(x)

        # create model
        model = tf.keras.Model(all_inputs, output)

        # compile model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=["accuracy"],
                      run_eagerly=True)

        return model

    def test_tensorflow_install(self):
        """"
        Test if tensorflow is installed correctly
        """
        print(tf.reduce_sum(tf.random.normal([1000, 1000])))


def main():
    model = ModelSetup()
    train_data = model.load_train_data()
    train, val, test = model.split_data(train_data)
    train_set = model.df_to_dataset(train, batch_size=5)
    val_set = model.df_to_dataset(train, batch_size=5)
    test_set = model.df_to_dataset(train, batch_size=5)
    [(train_features, label_batch)] = train_set.take(1)

    print('Every feature:', list(train_features.keys()))
    print('A batch of speed:', train_features['speed_in'])
    print('A batch of targets:', label_batch)

    speed_in_col = train_features['speed_in']
    layer = model.get_normalization_layer('speed_in', train_set)
    print(layer(speed_in_col))

    test_type_col = train_features['species']
    test_type_layer = model.get_category_encoding_layer(name='species',
                                                        dataset=train_set,
                                                        dtype='string')
    all_inputs, encoded_features = model.encode_features(train_set)
    ml_model = model.create_model(encoded_features, all_inputs)
    ml_model.fit(train_set, epochs=10, validation_data=val_set)
    result = ml_model.evaluate(test_set, return_dict=True)
    print(result)


if __name__ == "__main__":
    sys.exit(main())
