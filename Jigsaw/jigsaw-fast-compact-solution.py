# References
# This Kernel is heavily borrowing from several Kernels:
# [1] https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold  (most of it)
# [2] https://www.kaggle.com/artgor/cnn-in-keras-on-folds (preprocessing)
# [3] https://www.kaggle.com/taindow/simple-cudnngru-python-keras
# [4] https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480 (logger)
# [5] https://www.kaggle.com/thousandvoices/simple-lstm/ (embeddings)
##
import numpy as np 
import pandas as pd 
import os
import gc
import logging
import datetime
import warnings
from tqdm import tqdm
tqdm.pandas()
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
import keras.layers as L
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

COMMENT_TEXT_COL = 'comment_text'
EMB_MAX_FEAT = 300
MAX_LEN = 220
MAX_FEATURES = 100000
BATCH_SIZE = 1024
NUM_EPOCHS = 8
LSTM_UNITS = 64
NFOLDS = 4
EMB_PATHS = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]
JIGSAW_PATH = '../input/jigsaw-unintended-bias-in-toxicity-classification/'


def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
    
logger = get_logger()

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)

def build_embedding_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, EMB_MAX_FEAT))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
        except:
            embedding_matrix[i] = embeddings_index["unknown"]
            
    del embedding_index
    gc.collect()
    return embedding_matrix

def load_data():
    logger.info('Load train and test data')
    train = pd.read_csv(os.path.join(JIGSAW_PATH,'train.csv'), index_col='id')
    test = pd.read_csv(os.path.join(JIGSAW_PATH,'test.csv'), index_col='id')
    return train, test

def perform_preprocessing(train, test):
    logger.info('data preprocessing')
    
    # adding preprocessing from this kernel: https://www.kaggle.com/taindow/simple-cudnngru-python-keras
    punct_mapping = {"_":" ", "`":" "}
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct, mapping):
        for p in mapping:
            text = text.replace(p, mapping[p])    
        for p in punct:
            text = text.replace(p, f' {p} ')     
        return text

    for df in [train, test]:
        df[COMMENT_TEXT_COL] = df[COMMENT_TEXT_COL].astype(str)
        df[COMMENT_TEXT_COL] = df[COMMENT_TEXT_COL].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
    
    return train, test


def run_tokenizer(train, test):
    logger.info('Fitting tokenizer')
    tokenizer = Tokenizer(num_words=MAX_FEATURES, lower=True) 
    tokenizer.fit_on_texts(list(train[COMMENT_TEXT_COL]) + list(test[COMMENT_TEXT_COL]))
    word_index = tokenizer.word_index
    X_train = tokenizer.texts_to_sequences(list(train[COMMENT_TEXT_COL]))
    y_train = np.where(train['target'] >= 0.5, 1, 0)
    X_test = tokenizer.texts_to_sequences(list(test[COMMENT_TEXT_COL]))
    
    X_train = pad_sequences(X_train, maxlen=MAX_LEN)
    X_test = pad_sequences(X_test, maxlen=MAX_LEN)
    
    del tokenizer
    gc.collect()
    return X_train, X_test, y_train, word_index

def build_embeddings(word_index):
    logger.info('Load and build embeddings')
    embedding_matrix = np.concatenate(
        [build_embedding_matrix(word_index, f) for f in EMB_PATHS], axis=-1) 
    return embedding_matrix


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def build_model(embedding_matrix, word_index, verbose = False, compile = True):
    logger.info('Build model')
    sequence_input = L.Input(shape=(MAX_LEN,), dtype='int32')
    embedding_layer = L.Embedding(*embedding_matrix.shape,
                                weights=[embedding_matrix],
                                trainable=False)
    x = embedding_layer(sequence_input)
    x = L.SpatialDropout1D(0.2)(x)
    x = L.Bidirectional(L.CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    att = Attention(MAX_LEN)(x)
    avg_pool1 = L.GlobalAveragePooling1D()(x)
    max_pool1 = L.GlobalMaxPooling1D()(x)
    x = L.concatenate([att,avg_pool1, max_pool1])
    preds = L.Dense(1, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    if verbose:
        model.summary()
    if compile:
        model.compile(loss='binary_crossentropy',optimizer=Adam(0.005),metrics=['acc'])
    return model
    

def run_model(X_train, X_test, y_train, embedding_matrix, word_index):
    logger.info('Prepare folds')
    folds = KFold(n_splits=NFOLDS, random_state=42)
    oof_preds = np.zeros((X_train.shape[0]))
    sub_preds = np.zeros((X_test.shape[0]))
    
    logger.info('Run model')
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        
        K.clear_session()
        check_point = ModelCheckpoint(f'mod_{fold_}.hdf5', save_best_only = True)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=-1, patience=3)
        model = build_model(embedding_matrix, word_index)
        model.fit(X_train[trn_idx],
            y_train[trn_idx],
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            validation_data=(X_train[val_idx], y_train[val_idx]),
            callbacks = [early_stopping,check_point])
    
        oof_preds[val_idx] += model.predict(X_train[val_idx])[:,0]
        sub_preds += model.predict(X_test)[:,0]
    sub_preds /= folds.n_splits
    print(roc_auc_score(y_train,oof_preds))
    logger.info('Complete run model')
    return sub_preds

def submit(sub_preds):
    logger.info('Prepare submission')
    submission = pd.read_csv(os.path.join(JIGSAW_PATH,'sample_submission.csv'), index_col='id')
    submission['prediction'] = sub_preds
    submission.reset_index(drop=False, inplace=True)
    submission.to_csv('submission.csv', index=False)

def main():
    train, test = load_data()
    train, test = perform_preprocessing(train, test)
    X_train, X_test, y_train, word_index = run_tokenizer(train, test)
    embedding_matrix = build_embeddings(word_index)
    sub_preds = run_model(X_train, X_test, y_train, embedding_matrix, word_index)
    submit(sub_preds)
    
if __name__ == "__main__":
    main()
    