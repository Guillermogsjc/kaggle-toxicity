#%% SNNNs: es necesario Keras 2.0.5
from keras.layers import Input, Dense, merge, Reshape, Dropout, BatchNormalization
from keras.models import Model
from keras.constraints import maxnorm
from keras.regularizers import  l1_l2
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from keras.layers.noise import AlphaDropout
from keras.initializers import TruncatedNormal, RandomUniform
from keras.initializers import lecun_normal


def SNNN_binary_sig(input_shape_,
                    lista_layers,
                    lista_dropouts,
                    maxn,
                    optimizer_='Adam', lr=0.001):
    
    """ SNNNs para clasificacion 0-1 con salida sigmoide"""
    

    main_input = Input(shape=(input_shape_,), name='main_input')

    x_d = Dense(lista_layers[0],
                activation='selu',
                kernel_initializer='lecun_normal',
                kernel_constraint=maxnorm(maxn))(main_input)
    x = AlphaDropout(lista_dropouts[0])(x_d)

    for i, k in enumerate(lista_layers[1:]):
        x_d = Dense(k,
                    activation='selu',
                    kernel_constraint=maxnorm(maxn),
                    kernel_initializer='lecun_normal')(x)
        x = AlphaDropout(lista_dropouts[i + 1])(x_d)
    

    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[main_input], outputs=[main_output])

    model.compile(optimizer=optimizer_, loss=['binary_crossentropy'])

    return model




#%% Callback con roc_auc y pr_auc

from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, average_precision_score


class ExtraValMetric(Callback):
    def on_train_begin(self, logs={}):
        self.roc_aucs = []
        self.pr_aucs = []
        self.losses = []
        self.params['metrics'].append('val_roc_auc')
        self.params['metrics'].append('val_pr_auc')
        logs['val_roc_auc'] = 0
        logs['val_pr_auc'] = 0

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        self.losses.append(logs.get('loss'))
        
        y_pred = self.model.predict(self.validation_data[0])
        roc_auc_epoch = roc_auc_score(self.validation_data[1].ravel(), y_pred.ravel(), average='weighted')
        pr_auc_epoch = average_precision_score(self.validation_data[1].ravel(), y_pred.ravel(), average='weighted')
        logs['val_roc_auc'] = roc_auc_epoch
        logs['val_pr_auc'] = pr_auc_epoch
        self.roc_aucs.append(roc_auc_epoch)
        logs['val_roc_auc'] = roc_auc_epoch
        logs['val_pr_auc'] = pr_auc_epoch
        self.pr_aucs.append(pr_auc_epoch)
        
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return




