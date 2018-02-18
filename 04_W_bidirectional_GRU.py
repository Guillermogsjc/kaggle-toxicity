
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import constants as ct
from kaggletoxicity.keras_utils import ExtraValMetric, KaggleToxicityValMetric
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

max_features = 20000 #20000
maxlen = 500 # 200

train = pd.read_csv(os.path.join(ct.DATA_FOLDER, 'train.csv'))
test = pd.read_csv(os.path.join(ct.DATA_FOLDER, 'test.csv'))
# train = train.sample(frac=0.05, random_state=0)

list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("CVxTz").values

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, padding='pre', maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, padding='pre', maxlen=maxlen)


# In[ ]:


def get_bidirectional_model(embed_size,
                            input_shape,
                            n_neurons,
                            dropout_rate=0.1,
                            opt_alg='nadam'):
    inp = Input(shape=(input_shape,))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(GRU(n_neurons, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(n_neurons, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt_alg)

    return model


# In[ ]:


embedding_size = 128

model = get_bidirectional_model(embed_size=embedding_size,
                                input_shape=maxlen,
                                n_neurons=50, #50
                                dropout_rate=0.1)

batch_size = 1024
epochs = 50
val_prop = 0.2
es_patience = 5
rlr_patience = 2
rlr_cooldown = 4

file_path = os.path.join(ct.MODELS_FOLDER, "weights_base_best.hdf5")
extraval = KaggleToxicityValMetric()
early_stop = EarlyStopping(monitor='val_roc_auc', patience=es_patience, mode='max',  verbose=0)
checkpoint = ModelCheckpoint(file_path, monitor='val_roc_auc', verbose=0, mode='max',   save_best_only=True)
reduce_lr = ReduceLROnPlateau( monitor='val_roc_auc', 
                              factor=0.5, 
                              patience=rlr_patience, 
                              cooldown=rlr_cooldown, 
                              min_lr=1e-4)

callbacks_list = [extraval, checkpoint, early_stop, reduce_lr]
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=val_prop, callbacks=callbacks_list)


# In[ ]:


model.load_weights(file_path)

y_test = model.predict(X_te)

sample_submission = pd.read_csv(os.path.join(
    ct.DATA_FOLDER, 'sample_submission.csv'))

sample_submission[list_classes] = y_test

sample_submission.to_csv(os.path.join(ct.RESULTS_FOLDER, 'baseline.csv'))

