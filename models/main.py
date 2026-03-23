import pandas as pd
import numpy as np
from analytic import AnalyticLinearReg
from GradDescent import GDModel
from SGD import SGDModel


def main():
    if __name__ == '__main__':
        file_path = '~/LinearRegression/data/YearPredictionMSD.txt'
        df = pd.read_csv(file_path)

        # the train test split advised by dataset
        train = df[:463715]
        test = df[463715:]

        train_labels= train.iloc[:, 0]
        train_features = train.iloc[:, 1:]
        mean = np.mean(train_features, axis=0)
        std = np.std(train_features, axis=0)
        std[std == 0] = 1.0

        #normalize data
        train_features = (train_features - mean) / std

        test_labels = test.iloc[:, 0]
        test_features = test.iloc[:, 1:]

        test_features = (test_features - mean) / std


        # add the column of ones
        ones_train = np.ones((train_features.shape[0], 1))
        train_features = np.concatenate([ones_train, train_features], axis=1)
        ones_test = np.ones((test_features.shape[0], 1))
        test_features = np.concatenate([ones_test, test_features], axis=1)

        analyticModel = AnalyticLinearReg()
        analyticModel.fit(train_features, train_labels)

        print(f'MSE of analytic model: {analyticModel.evaluate(test_features, test_labels)}')
        

        gd = GDModel(test_features.shape[1])
        cnt = gd.fit(train_features, train_labels, 0.001)
        print(
            f'MSE of simple gradient descent:\
            {gd.evaluate(test_features, test_labels)}\
            number of iterations: {cnt}'
        )

        sgd = SGDModel(test_features.shape[1])
        cnt = sgd.fit(train_features, train_labels, 0.001)
        print(
            f'MSE of stochastic gradient descent:\
            {sgd.evaluate(test_features, test_labels)}\
            number of iterations: {cnt}'
        )
main()