#https://www.youtube.com/watch?v=-vAgZpyfv40
#https://github.com/vprusso/youtube_tutorials/blob/master/machine_learning/text_classification/imdb.py
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
imdb = keras.datasets.imdb

import numpy as np
# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# restore np.load for future normal usage
np.load = np_load_old


print(f"Training entries {len(train_data)}. Labels: {len(train_labels)}")


#print(len(train_data[0]), len(train_data[1]))

word_index = imdb.get_word_index()

word_index = {k: (v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# decoding, turning the list of integers into paragraph
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, "?") for i in text])


print(train_data[0])
print(decode_review(train_data[0]))

# padding so that the movie reviews  has the same length for the network
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding="post",
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding="post",
                                                       maxlen=256)


print(len(train_data[0]), len(train_data[1]))

print(train_data[0])

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 48))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

#model.summary()

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["acc"])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=2,
                    batch_size=512,
                    validation_data=(x_val,y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)