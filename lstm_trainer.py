import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv1D, Bidirectional, LSTM, AveragePooling1D
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import imblearn.over_sampling
import numpy as np

from signalpreparator import SignalPreparator


class LstmTrainer:

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
        sp = SignalPreparator(3, 360)
        sp.abnormal_beat_annotations = ['V']
        records = [line.rstrip('\n') for line in open(self.data_path + 'RECORDS.txt')]
        records.remove('101')  # these records have approx 2500 normal beats and up to 3 abnormal
        records.remove('103')
        records.remove('117')
        records.remove('220')
        records.remove('230')

        rec_train = []
        rec_valid = []

        # take random subsets for train and validation sets but make sure that the distribution normal vs abnormal
        # is similar in both sets
        iter_count = 0
        while True:
            if iter_count > 1000:
                raise Exception(
                    "Could not find sets of samples where ratios between validation and train sets would be satisfactory")
            rec_train = random.sample(records, int((1 - self.validation_ratio) * len(records)))
            rec_valid = [rec for rec in records if rec not in rec_train]
            # print("Train set size: %d, Validation set size %d" % (len(rec_train), len(rec_valid)))

            normal, abnormal, others = sp.get_beat_type_distribution(rec_train, False)
            normalV, abnormalV, othersV = sp.get_beat_type_distribution(rec_valid, False)

            # if difference in normal vs abnormal distribution between train and validation sets is larger than 5%
            # take another random sample
            if abs(abnormal / (normal + abnormal) - abnormalV / (abnormalV + normalV)) < 0.05:
                print("\nTrain set size: %d records, Validation set size %d records" % (len(rec_train), len(rec_valid)))
                print("Train set includes %d normal and %d abnormal beats (ratio abnormal to all is %5f)" % (
                normal, abnormal, abnormal / (normal + abnormal)))
                print("Validation set includes %d normal and %d abnormal beats (ratio abnormal to all is %5f)" % (
                normalV, abnormalV, abnormalV / (normalV + abnormalV)))
                print("(Note that the actual number is a little smaller because some of these beats are on the edge and are omitted)\n")
                break
            iter_count += 1
        # end of while

        x_train, y_train, _ = sp.load_dataset(rec_train)
        x_valid, y_valid, _ = sp.load_dataset(rec_valid)

        # Removing some normal beats to reduce complexity
        print("Train set reduced from", len(y_train), "to ", end='')

        train_normal_indices = np.argwhere(y_train.flatten() == 0).flatten()  # indices that point to normal beats
        clip = int((len(train_normal_indices) - len(y_train) / 2) * 2)
        if clip > 0:
            train_normal_indices = np.random.choice(train_normal_indices, clip, replace=False)
            train_new_mask = np.ones(y_train.shape[0])
            train_new_mask[train_normal_indices] = 0
            train_new_mask = np.argwhere(train_new_mask == 1).flatten()

            x_train = x_train[train_new_mask, :]
            y_train = y_train[train_new_mask]

        print(len(y_train))
        print("Validation set reduced from", len(y_valid), "to ", end='')
        valid_normal_indices = np.argwhere(y_valid.flatten() == 0).flatten()  # indices that point to normal beats
        clip = int((len(valid_normal_indices) - len(y_valid) / 2) * 2)
        if clip > 0:
            valid_normal_indices = np.random.choice(valid_normal_indices, clip, replace=False)
            valid_new_mask = np.ones(y_valid.shape[0])
            valid_new_mask[valid_normal_indices] = 0
            valid_new_mask = np.argwhere(valid_new_mask == 1).flatten()

            x_valid = x_valid[valid_new_mask, :]
            y_valid = y_valid[valid_new_mask]

        print(len(y_valid))

        train_value, train_count = np.unique(y_train, return_counts=True)
        print("Train set now includes %d %s and %d %s" % (train_count[0], 'N' if train_value[0] == 0 else 'A', train_count[1], 'N' if train_value[1] == 0 else 'A'))
        valid_value, valid_count = np.unique(y_valid, return_counts=True)
        print("Valid set now includes %d %s and %d %s" % (valid_count[0], 'N' if valid_value[0] == 0 else 'A', valid_count[1], 'N' if valid_value[1] == 0 else 'A'))


        # if self.over_sampling_method == 'adasyn':
        #    ada = imblearn.over_sampling.SMOTENC(random_state=42, categorical_features=[])
        #    x_train_cnn, y_train = ada.fit_resample(X=x_train.T, y=y_train.T)

        # X_train, y_train = ada.fit_resample(
        #    X=X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2])), y=y_train)
        # X_train = X_train.reshape((X_train.shape[0], 112, 112, 1))

        x_train_cnn = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_valid_cnn = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))

        # LEARNING
        model = Sequential()
        model.add(Bidirectional(LSTM(64, input_shape=(x_train_cnn.shape[1], x_train_cnn.shape[2]))))
        model.add(Dropout(rate=0.25))
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


nn = LstmTrainer('data/', 0.25)
nn.train()
