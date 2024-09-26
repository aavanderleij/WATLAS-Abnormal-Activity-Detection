"""
setup for watlas disturbance model
"""
import sys

import numpy as np
import pandas as pd

import tensorflow as tf

print(tf.__version__)
from tensorflow.keras import layers


class ModelSetup:

    def __init__(self):
        # TODO add  exact species per group
        # TODO add time of day/time to high tide when more training data is available
        # TODO add tidal data
        # columns by type
        self.columns_to_drop = ["df_id", "time_agg", "tag", "posID", "TAG", "large_disturb", "small_disturb", "time",
                                "TIME", "large_disturb_1", "large_disturb_2", "large_disturb_3", "large_disturb_4",
                                'small_disturb_1', 'small_disturb_2', 'small_disturb_3', 'small_disturb_4', "speed_out"]
        self.numerical_columns = ['NBS', 'speed_in', 'VARX', 'VARY', 'group_size_per_timestamp',
                                  'n_unique_species', 'avg_dist', 'std_dist', 'distance', 'speed', 'turn_angle',
                                  'speed_1', 'distance_1', 'turn_angle_1', 'speed_2', 'distance_2', 'turn_angle_2',
                                  'speed_3', 'distance_3', 'turn_angle_3', 'speed_4', 'distance_4', 'turn_angle_4']
        self.categorical_sting_columns = ["species"]
        self.categorical_bool_columns = ["alert", 'alert_1', 'alert_2', 'alert_3', 'alert_4']

    def load_train_data(self, path_to_traindata="data/training_data/processed_train_data.csv"):
        train_data_df = pd.read_csv(path_to_traindata)
        # drop id data
        # TODO keep an eye on parameters
        train_data_df = train_data_df.drop(columns=self.columns_to_drop)
        print(train_data_df.isna().sum())  # Check for NaNs
        print(train_data_df.isin([np.inf, -np.inf]).sum())
        return train_data_df

    def split_data(self, dataframe):
        train, val, test = np.split(dataframe.sample(frac=1), [int(0.8 * len(dataframe)), int(0.9 * len(dataframe))])
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
            ds: TensorFlow dataset
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
        # Create a Normalization layer for the feature.
        normalizer = layers.Normalization(axis=None)

        # Prepare a Dataset that only yields the feature.
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)

        return normalizer

    def get_category_encoding_layer(self, name, dataset, dtype, max_tokens=None):
        # Create a layer that turns strings into integer indices.
        if dtype == 'string':
            index = layers.StringLookup(max_tokens=max_tokens)
        # Otherwise, create a layer that turns integer values into integer indices.
        else:
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
        all_inputs = {}
        encoded_features = []

        # Numerical features.
        for header in self.numerical_columns:
            numeric_col = tf.keras.Input(shape=(1,), name=header)
            normalization_layer = self.get_normalization_layer(header, dataset)
            encoded_numeric_col = normalization_layer(numeric_col)
            all_inputs[header] = numeric_col
            encoded_features.append(encoded_numeric_col)

        for header in self.categorical_sting_columns:
            categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
            encoding_layer = self.get_category_encoding_layer(name=header,
                                                              dataset=dataset,
                                                              dtype='string',
                                                              max_tokens=9)
            encoded_categorical_col = encoding_layer(categorical_col)
            all_inputs[header] = categorical_col
            encoded_features.append(encoded_categorical_col)

        for header in self.categorical_bool_columns:
            bool_col = tf.keras.Input(shape=(1,), name=header)
            all_inputs[header] = bool_col
            encoded_features.append(bool_col)

        return all_inputs, encoded_features

    def create_model(self, encoded_features, all_inputs):
        all_features = tf.keras.layers.concatenate(encoded_features)
        x = tf.keras.layers.Dense(32, activation="relu")(all_features)
        x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(all_inputs, output)

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=["accuracy"],
                      run_eagerly=True)

        return model

    def test_tensorflow_install(self):
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
