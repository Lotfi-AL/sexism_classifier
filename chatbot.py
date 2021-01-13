import json
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import config
from tensorflow.keras.preprocessing import sequence
import numpy as np
import os
import argparse
import data
from preprocess import preprocess_sentence
import matplotlib.pyplot as plt
import sys


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()


def make_model():
    # +1 for padding size.
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(config.VOCAB_SIZE+1, 256),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.LSTM(512),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=['accuracy'])
    return model


checkpoint_dir = './training_checkpoints'


def train_latest_checkpoint():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(
        physical_devices[0], True)
    train_data, test_data, train_labels, test_labels = make_train_test_split()

    model = tf.keras.models.load_model(
        checkpoint_dir)
    history = model.fit(train_data, train_labels, epochs=40, validation_split=0.1,
                        callbacks=[make_callbacks()], batch_size=32, shuffle=True)

    nb_epoch = len(history.history['loss'])
    learning_rate = history.history['lr']
    xc = range(nb_epoch)
    plt.plot(xc, learning_rate)
    plt.xlabel('num of Epochs')
    plt.ylabel('learning rate')
    plt.title('Learning rate')
    plt.grid(True)
    plt.style.use(['seaborn-ticks'])
    print('\n# Evaluate on test data')
    results = model.evaluate(test_data, test_labels)
    print('test loss, test acc:', results)

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')


def make_train_test_split():
    test_split = 0.9
    train_data = data.load_enc_ids()
    train_data = sequence.pad_sequences(train_data, config.MAXLEN)
    train_data_length = len(train_data)

    test_data = train_data[int(train_data_length*test_split):]
    train_data = train_data[:int(train_data_length*test_split)]

    train_labels = data.load_dec_ids()
    train_labels = np.array(train_labels)
    test_labels = train_labels[int(train_data_length*test_split):]
    train_labels = train_labels[:int(train_data_length*test_split)]
    print(train_labels)
    print(train_data.shape)
    return train_data, test_data, train_labels, test_labels


def make_callbacks():
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                                     patience=5, min_lr=0.00001)

    # Directory where the checkpoints will be saved
    # checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir, monitor="val_loss", save_best_only=True, mode="min"
    )

    json_log = open('loss_log.json', mode='wt', buffering=1)
    json_logging_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: json_log.write(
            json.dumps({'epoch': str(epoch), 'loss': str(logs['lr'])}) + '\n'),
        on_train_end=lambda logs: json_log.close()
    )

    return [reduce_lr, json_logging_callback, checkpoint_callback]


def train():
    # early_stopping = tf.keras.callbacks.EarlyStopping()
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # config = tf.config.experimental.set_memory_growth(
    #     physical_devices[0], True)
    train_data, test_data, train_labels, test_labels = make_train_test_split()
    model = make_model()

    history = model.fit(train_data, train_labels, epochs=40, validation_split=0.1,
                        callbacks=[make_callbacks()], batch_size=32, shuffle=True, initial_epoch=0)

    nb_epoch = len(history.history['loss'])
    learning_rate = history.history['lr']
    xc = range(nb_epoch)
    plt.plot(xc, learning_rate)
    plt.xlabel('num of Epochs')
    plt.ylabel('learning rate')
    plt.title('Learning rate')
    plt.grid(True)
    plt.style.use(['seaborn-ticks'])
    print('\n# Evaluate on test data')
    results = model.evaluate(test_data, test_labels)
    print('test loss, test acc:', results)

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')


def pad_sentence(words):
    return sequence.pad_sequences(words, config.MAXLEN)


def process(words):
    words = data.words_to_ids(words)
    return np.array(words)


def from_test_data_model(model):
    train_data, test_data, train_labels, test_labels = make_train_test_split()
    model.evaluate(test_data, test_labels)


def from_file_data_model(enc_file, dec_file, model):
    test_enc = []
    test_dec = []
    test_data_tweet = []
    test_labels_tweet = []

    with open("preprocessed/"+enc_file) as file:
        for line in file:
            test_enc.append(line.strip())

    with open("preprocessed/"+dec_file) as file:
        for line in file:
            test_dec.append(line.strip())

    for item in test_enc:
        test_data_tweet.append(process(item))

    test_labels_tweet = np.array(data.labels_to_ids(test_dec))
    test_data_tweet = sequence.pad_sequences(test_data_tweet, config.MAXLEN)
    r2 = model.evaluate(test_data_tweet, test_labels_tweet)
    ls = []
    for item in model.predict(test_data_tweet):
        print(item)
    # print(ls)
    # print(model.predict(test_data_tweet))


def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


def from_input(model):
    print("Welcome to tweet classifier, type in a tweet and i will classify either, non-offensive or sexist. Max length is: ", config.MAXLEN)
    with open("preprocessed/output.txt", "a+") as output_file:
        while True:
            line = _get_user_input()
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]
            if line == '':
                break
            output_file.write("HUMAN ++++"+line+"\n")
            r = []
            r.append(process(line.strip()))
            # r.append(line.strip())
            token_ids = sequence.pad_sequences(r, config.MAXLEN)
            print(model.predict(token_ids))
            response = data.id_to_label(model.predict(token_ids))
            print(response)
            output_file.write("BOT ++++" + response + "\n")

        output_file.write('=============================================\n')


def tree_labels(model):
    train_data, test_data, train_labels, test_labels = make_train_test_split()
    print("hey", model.evaluate(test_data, test_labels))
    # model.


def chat():
    model = tf.keras.models.load_model(
        config.checkpoint_dir)
    # train_data, test_data, train_labels, test_labels = make_train_test_split()
    from_file_data_model("test.enc", "test.dec", model)
    # from_test_data_model(model)
    # tree_labels(model)
    # from_input(model)
    # from_input(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat', 'cp'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'chat':
        chat()
    elif args.mode == 'cp':
        train_latest_checkpoint()


if __name__ == '__main__':
    main()
