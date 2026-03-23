import pandas as pd
from analytic import AnalyticLinearReg
from GradDescent import GDModel


def main():
    if __name__ == '__main__':
        file_path = '~/LinearRegression/data/YearPredictionMSD.txt'
        df = pd.read_csv(file_path)

        # the train test split advised by dataset
        train = df[:463715]
        test = df[463715:]

        train_labels= train.iloc[:, 0]
        train_features = train.iloc[:, 1:]

        test_labels = test.iloc[:, 0]
        test_features = test.iloc[:, 1:]

        analyticModel = AnalyticLinearReg()
        analyticModel.fit(train_features, train_labels)


        print(f'MSE of analytic model: {analyticModel.evaluate(test_features, test_labels)}')
        

        gd = GDModel(test_features.shape[0], test_features.shape[1])
        gd.fit(train_features, train_labels, 0.01)
        print(f'MSE of simple gradient descent: {gd.evaluate(test_features, test_labels)}')

main()