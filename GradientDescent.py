import numpy as np
import pandas as pd


# x is the features matrix, y is the outputs, alpha is the step and n is the number of iterations to run
def gradient_descent(x, y, alpha, n):
    errors = []
    y = y.reshape(len(y), 1)
    m = (len(x[0]))
    theta = np.random.rand(m, 1)
    for i in range(n):
        h = np.matmul(x, theta)
        error = np.mean(np.square(h - y))
        errors.append(error)
        # error.append(np.square(h - y).mean(axis=0))
        d = np.matmul(np.transpose(h-y), x)
        theta = theta - np.transpose(d)*alpha*1/m

    return theta, errors


def main():
    data = pd.read_csv("Traffic_Volume.csv")
    data = data.drop(['holiday', 'weather_main', 'weather_description', 'date_time'], axis=1)
    x = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    x = (x-np.min(x))/(np.max(x)-np.min(x))
    theta, error = gradient_descent(x, y, .001, 10000)
    print(error)
    print(theta)


if __name__ == '__main__':
    main()

