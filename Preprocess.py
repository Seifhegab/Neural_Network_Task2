import pandas as pd
from sklearn import preprocessing


class Preprocess:
    data = None

    def __init__(self,data):
        self.data = data

    def fill_null(self):
        # get the mean of MinorAxisLength
        MinorAxisLengthMean = self.data["MinorAxisLength"].mean()
        # Fill nulls
        self.data["MinorAxisLength"].fillna(MinorAxisLengthMean, inplace=True)
        self.data["Area"].fillna(MinorAxisLengthMean, inplace=True)
        self.data["Perimeter"].fillna(MinorAxisLengthMean, inplace=True)
        self.data["MajorAxisLength"].fillna(MinorAxisLengthMean, inplace=True)
        self.data["roundnes"].fillna(MinorAxisLengthMean, inplace=True)

    def Normalize(self):
        # normalize the data between -1,1
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        Bean_Data1 = scaler.fit_transform(self.data)
        # change data to dataframe
        Bean_Data1 = pd.DataFrame(Bean_Data1, columns=self.data.columns)
        return Bean_Data1