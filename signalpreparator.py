import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import lfilter


class SignalPreparator:
    abnormal_beat_annotations = ['L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f',
                                 'Q', '?']
    non_beat_annotations = ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', '\'', '^', '|', '~', '+', 's', 'T',
                            '*', 'D',
                            '=', '"', '@']

    def __init__(self, surrounding_seconds, freq):
        self.validation_ratio = 0.25
        self.data_path = 'data/'
        random.seed(42)
        self.surrounding_seconds = surrounding_seconds
        self.freq = freq

    def load_signal(self, record_number):
        """ Loads ECG signal with given record_number.

                Parameters
                ----------
                record_number : str
                    Number of signal to be loaded.

                Returns
                ----------
                p_signal : list
                    List of signal values at sampled intervals. If there are multiple signals each value is a list of values.
                atr_sym : list
                    List of annotations at times specified in atr_sample
                atr_sample : list
                    Times of annotations in atr_sym (time values are consecutive sample numbers)

        """
        file = self.data_path + record_number
        record = wfdb.rdrecord(file)
        annotation = wfdb.rdann(file, 'atr')
        p_signal = record.p_signal

        assert record.fs == self.freq, 'Record frequency is not %dHz' % self.freq

        atr_sym = annotation.symbol
        atr_sample = annotation.sample

        return p_signal, atr_sym, atr_sample

    def plot_signal(self, signal):
        plt.plot(signal)
        plt.show()

    def filter_signal(self, signals):
        """Applies filters on given signals. There may be any number of signals provided as rows of the 2D array.
        All of them are filtered by low band pass filter, then notch filter and then by derivative filter.
        For merging absolute values of filtered signals are summed.

        Parameters
        ----------
        signals : np.array
            Array of signals. Each represent a different signal.
        Returns
        -------
        Filtered and merged signal.
        """
        final_signal = np.zeros(signals.shape[0])

        for signal in signals.T:
            new_signal = self.filter1(signal)
            new_signal = self.filter2(new_signal)
            new_signal = self.filter3(new_signal)

            final_signal += np.abs(new_signal)
        return final_signal

    def filter1(self, signal):
        a = 4
        b = np.array([1, 2, 1])
        return lfilter(b, a, signal)

    def filter2(self, signal):
        a = 1
        b = [1, -2 * np.cos(60 * np.pi / 125), 1]
        return lfilter(b, a, signal)

    def filter3(self, signal):
        a = 1
        b = [1, 0, 0, 0, 0, 0, - 1]
        return lfilter(b, a, signal)

    def load_dataset(self, list_of_records=None):
        """Creates pandas dataframe for read records.

        Parameters
        ---------
        list_of_records : list or None
            List of numbers of signal to be loaded. They can start with paths.
            If empty or none, all found record will be loaded.

        Returns
        -------
        X_all : np.ndarray
            An array of short clips of signal that are centered on beat and are surrounded by self.surrounding_seconds seconds
            long subsignal. Each clip as represented as an array of values.
        Y_all : ndarray
            An array with values 0 or 1, which represent the class (normal or abnormal) of a sample clip.
        sym_all : list
            A list of strings that represent classes of sample clips.
       """
        records = [line.rstrip('\n') for line in open(self.data_path + 'RECORDS.txt')]

        if list_of_records is not None and len(list_of_records) > 0:
            records = list(set(records) & set(map(lambda rec: str(rec), list_of_records)))

        if len(records) == 0:
            raise Exception("Intersection of requested records and actual ones is empty")
        print("Creating a dataset from %d records" % len(records))
        
        # makes dataset ignoring non-beats
        num_cols = 2 * self.surrounding_seconds * self.freq
        X_all = np.zeros((1, num_cols))
        Y_all = np.zeros((1, 1))
        sym_all = []

        # list that keeps track of numbers of beats per patient
        max_rows = []

        for rec in records:
            p_signal, atr_sym, atr_sample = self.load_signal(rec)

            # apply digital filter on the signal
            p_signal = self.filter_signal(p_signal)
            df_ann = pd.DataFrame({'atr_sym': atr_sym, 'atr_sample': atr_sample})
            # remove non-beat annotations
            df_ann = df_ann.loc[df_ann.atr_sym.isin(self.abnormal_beat_annotations + ['N'])]

            # loop through acceptable beats
            num_rows = len(df_ann)

            X = np.zeros((num_rows, num_cols))
            Y = np.zeros((num_rows, 1))
            max_row = 0

            # fills X with sample clips and Y with their corresponding classes
            for atr_sample, atr_sym in zip(df_ann.atr_sample.values, df_ann.atr_sym.values):
                left = max([0, (atr_sample - self.surrounding_seconds * self.freq)])
                right = min([len(p_signal), (atr_sample + self.surrounding_seconds * self.freq)])
                x = p_signal[left: right]

                if len(x) == num_cols:
                    X[max_row, :] = x
                    Y[max_row, :] = int(atr_sym in self.abnormal_beat_annotations)
                    sym_all.append(atr_sym)
                    max_row += 1

            X = X[:max_row, :] # removes unused rows
            Y = Y[:max_row, :]
            max_rows.append(max_row)
            X_all = np.append(X_all, X, axis=0)
            Y_all = np.append(Y_all, Y, axis=0)

        # drop the first zero row
        X_all = X_all[1:, :]
        Y_all = Y_all[1:, :]

        assert np.sum(max_rows) == X_all.shape[0], 'num of rows messed up'
        assert Y_all.shape[0] == X_all.shape[0], 'num of rows messed up'
        assert Y_all.shape[0] == len(sym_all), 'num of rows messed up'

        # Print distribution
        values, counts = np.unique(np.array(sym_all), return_counts=True)
        sum_abnormal = 0
        normal = 0
        for value, count in zip(values, counts):
            if value == 'N':
                normal = count
            else:
                print("%s: %5d" % (value, count))
                sum_abnormal += count
        print("%s: %d" % ("Abnormal", sum_abnormal))
        print("%s: %d" % ("Normal", normal))

        return X_all, Y_all, sym_all



#sp = SignalPreparator(3,360)
#X_all, Y_all, sym_all = sp.load_dataset([100, 101])
