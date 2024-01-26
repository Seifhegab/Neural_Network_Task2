from sklearn.model_selection import train_test_split
import numpy as np


class SplittingData:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def split(self):

        X_train1, X_test1, y_train1, y_test1 = train_test_split(self.x.iloc[0:50], self.y.iloc[0:50],
                                                                test_size=0.4, shuffle=True,random_state=10)

        X_train1 = np.array(X_train1)
        X_test1 = np.array(X_test1)
        y_train1 = np.array(y_train1)
        y_test1 = np.array(y_test1)

        X_train2, X_test2, y_train2, y_test2 = train_test_split(self.x.iloc[50:100], self.y.iloc[50:100],
                                                                test_size=0.4, shuffle=True, random_state=10)

        X_train2 = np.array(X_train2)
        X_test2 = np.array(X_test2)
        y_train2 = np.array(y_train2)
        y_test2 = np.array(y_test2)

        X_train3, X_test3, y_train3, y_test3 = train_test_split(self.x.iloc[100:150], self.y.iloc[100:150],
                                                                test_size=0.4, shuffle=True, random_state=10)

        X_train3 = np.array(X_train3)
        X_test3 = np.array(X_test3)
        y_train3 = np.array(y_train3)
        y_test3 = np.array(y_test3)

        X_train = np.concatenate((X_train1, X_train2, X_train3))
        X_test = np.concatenate((X_test1, X_test2, X_test3))
        y_train = np.concatenate((y_train1, y_train2, y_train3))
        y_test = np.concatenate((y_test1, y_test2, y_test3))

        return X_train,y_train,X_test,y_test