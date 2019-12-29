import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv1D, Bidirectional, LSTM, AveragePooling1D
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import imblearn.over_sampling
import numpy as np

from signalpreparator import SignalPreparator


class CnnTrainer:

    def __init__(self, data_dir, validation_ratio):
        if 0 <= validation_ratio >= 1:
            raise ValueError("You are an idiot")

        self.data_path = data_dir
        self.validation_ratio = validation_ratio

    def print_report(self, y_actual, y_pred, thresh):
        auc = roc_auc_score(y_actual, y_pred)
        accuracy = accuracy_score(y_actual, (y_pred > thresh))
        recall = recall_score(y_actual, (y_pred > thresh))
        precision = precision_score(y_actual, (y_pred > thresh))

        print("AUC: %.6f" % auc)
        print("Accuracy: %.6f" % accuracy)
        print("Recall: %.6f" % recall)
        print("Precision: %.6f" % precision)

    def train(self):
        sp = SignalPreparator(2, 360)
        records = [line.rstrip('\n') for line in open(self.data_path + 'RECORDS.txt')]
        records.remove('101')
        records.remove('103')
        records.remove('117')
        records.remove('230')

        rec_train = []
        rec_valid = []
        # take random subsets for train and validation sets but make sure that the distribution normal vs abnormal
        # is similar in both sets
        iter_count = 0
        while True:
            if iter_count > 1000:
                raise Exception("Could not find sets of samples where ratios between validation and train sets would be satisfactory")
            rec_train = random.sample(records, int((1 - self.validation_ratio) * len(records)))
            rec_valid = [rec for rec in records if rec not in rec_train]
            # print("Train set size: %d, Validation set size %d" % (len(rec_train), len(rec_valid)))

            normal, abnormal, others = sp.get_beat_type_distribution(rec_train, False)
            normalV, abnormalV, othersV = sp.get_beat_type_distribution(rec_valid, False)

            # if difference in normal vs abnormal distribution between train and validation sets is larger than 5%
            # take another random sample
            if abs(abnormal/(normal+abnormal) - abnormalV/(abnormalV+normalV)) < 0.05:
                print("Train set size: %d records, Validation set size %d records" % (len(rec_train), len(rec_valid)))
                print("Train set includes %d normal and %d abnormal beats (ratio abnormal to all is %5f)" % (normal, abnormal, abnormal/(normal+abnormal)))
                print("Validation set includes %d normal and %d abnormal beats (ratio abnormal to all is %5f)" % (normalV, abnormalV, abnormalV/(normalV+abnormalV)))
                break
            iter_count += 1
        # end of while


        x_train, y_train, sym_train = sp.load_dataset(rec_train)
        x_valid, y_valid, sym_valid = sp.load_dataset(rec_valid)

        #if self.over_sampling_method == 'adasyn':
        #    ada = imblearn.over_sampling.SMOTENC(random_state=42, categorical_features=[])
        #    x_train_cnn, y_train = ada.fit_resample(X=x_train.T, y=y_train.T)

           #X_train, y_train = ada.fit_resample(
           #    X=X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2])), y=y_train)
           #X_train = X_train.reshape((X_train.shape[0], 112, 112, 1))

        x_train_cnn = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_valid_cnn = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))
        print(x_train_cnn.shape)

        # for 3 seconds -> 2160, 1078, 537
        # for 2 seconds -> 1440, 718, 357
        # for 1 second -> 720, 358, 177

        # LEARNING
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(1440, 1)))
        model.add(Dropout(rate=0.5))
        model.add(AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(718, 128)))
        model.add(Dropout(rate=0.5))
        model.add(AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(357, 128)))
        model.add(Dropout(rate=0.5))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        model.fit(x_train_cnn, y_train, batch_size=32, epochs=2, verbose=1)

        y_train_preds_cnn = model.predict_proba(x_train_cnn, verbose=1)
        y_valid_preds_cnn = model.predict_proba(x_valid_cnn, verbose=1)

        thresh = (sum(y_train) / len(y_train))[0]
        print("thresh:", thresh)
        print('Train')
        self.print_report(y_train, y_train_preds_cnn, thresh)
        print('Valid')
        self.print_report(y_valid, y_valid_preds_cnn, thresh)


nn = CnnTrainer('data/', 0.25)
nn.train()
