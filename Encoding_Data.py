from sklearn import preprocessing
import pandas as pd


class EncodingData:
    def __init__(self, y):
        self.y = y

    def label_encode(self):
        Label_encoder = preprocessing.LabelEncoder()
        self.y = Label_encoder.fit_transform(self.y)
        self.y = pd.DataFrame({'Class': self.y})
        return self.y