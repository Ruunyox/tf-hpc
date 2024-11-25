import keras
import numpy as np
from datetime import datetime


class BasicRunStats(keras.callbacks.Callback):
    def __init__(self):
        super(BasicRunStats, self).__init__()
        self.start_time = None
        self.end_time = None
        self.train_start_time = None
        self.train_end_time = None
        self.train_epoch_times = []

    def on_epoch_begin(self, *args, **kwargs):
        self.train_start_time = datetime.now()

    def on_epoch_end(self, *args, **kwargs):
        self.train_end_time = datetime.now()
        duration = self.train_end_time.timestamp() - self.train_start_time.timestamp()
        self.train_epoch_times.append(duration)

    def on_train_begin(self, *args, **kwargs):
        self.start_time = datetime.now()

    def on_train_end(self, *args, **kwargs):
        self.end_time = datetime.now()
        duration = self.end_time.timestamp() - self.start_time.timestamp()
        avg_train = np.average(self.train_epoch_times)
        std_train = np.std(self.train_epoch_times)
        print("")
        print(">>> TRAINING SUMMARY <<<")
        print("===========================================================")
        print(f"total time       :  {duration:.4f}")
        print(f"avg train epoch  :  {avg_train:.4f} +/- {std_train:.4f}")
        print("===========================================================")
        print("")
