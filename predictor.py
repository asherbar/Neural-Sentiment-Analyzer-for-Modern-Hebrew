import csv
import os

from keras import Input, Model, optimizers
from keras.layers import Embedding, Conv1D, MaxPool1D, Concatenate, Flatten, Dense, Dropout
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras_preprocessing import text


class Predictor:
    def __init__(self, train_data_path='data/token_train.tsv', max_document_length=100, vocabulary_size=5000,
                 embedding_size=300,
                 dropout_keep_prob=0.5, lr=1e-4, batch_size=50, num_epochs=5, dev_size=0.2):
        self.dev_size = dev_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout_keep_prob = dropout_keep_prob
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.max_document_length = max_document_length
        self.train_data_path = train_data_path
        self.pickle_path = os.path.splitext(train_data_path)[0] + '.model.p'
        self._tokenizer = text.Tokenizer(num_words=vocabulary_size, char_level=False, filters='')

        raw_x, raw_y = self._aggregate_raw_data_from_dir(self.train_data_path)
        self._tokenizer.fit_on_texts(raw_x)

        self.x_train, self.y_train = self._prepare_from_raw_data(raw_x, raw_y)
        self.model = self._create_model()

    def train(self):
        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.num_epochs, verbose=1,
                       validation_split=self.dev_size)

    def evaluate(self, test_data_path='data/token_test.tsv'):
        x_token_test, y_token_test = self._prepare_test_data(test_data_path)
        return self.model.evaluate(x_token_test, y_token_test, batch_size=self.batch_size, verbose=1)

    @classmethod
    def _aggregate_raw_data_from_dir(cls, dir_path):
        raw_x, raw_y = [], []
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(".tsv"):
                print(f'Adding training data from {filename}')
                tsv_path = os.path.join(dir_path, filename)
                file_raw_x, file_raw_y = cls._raw_data_from_tsv(tsv_path)
                raw_x.extend(file_raw_x)
                raw_y.extend(file_raw_y)
        print(f'Training data length: {len(raw_x)}')
        return raw_x, raw_y

    @classmethod
    def _raw_data_from_tsv(cls, tsv_path):
        raw_x, raw_y = [], []
        with open(tsv_path) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                raw_x.append(row[0])
                raw_y.append(row[1])
        return raw_x, raw_y

    def _prepare_test_data(self, test_data_path):
        raw_x, raw_y = self._raw_data_from_tsv(test_data_path)
        return self._prepare_from_raw_data(raw_x, raw_y)

    def _prepare_from_raw_data(self, raw_x, raw_y):
        tokenized_x = self._tokenizer.texts_to_sequences(raw_x)
        padded_x = sequence.pad_sequences(tokenized_x, maxlen=self.max_document_length, padding='post',
                                          truncating='post')
        categorical_y = to_categorical(raw_y, 3)
        return padded_x, categorical_y

    def _create_model(self):
        convs = []
        text_input = Input(shape=(self.max_document_length,))
        x = Embedding(self.vocabulary_size, self.embedding_size)(text_input)
        for fsz in [3, 8]:
            conv = Conv1D(128, fsz, padding='valid', activation='relu')(x)
            pool = MaxPool1D()(conv)
            convs.append(pool)
        x = Concatenate(axis=1)(convs)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout_keep_prob)(x)
        preds = Dense(3, activation='softmax')(x)
        model = Model(text_input, preds)
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=self.lr), metrics=['accuracy'])
        return model


def main():
    predictor = Predictor(train_data_path='data/training')
    predictor.train()
    scores = predictor.evaluate('data/headline_1.tsv')
    print('\nAccuracy: {:.3f}'.format(scores[1]))


if __name__ == '__main__':
    main()
