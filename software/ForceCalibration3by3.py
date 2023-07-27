import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":
    data_path = 'path_to_data'
    all_data = pd.read_csv(data_path)

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    sns.pairplot(all_data, x_vars=['Fx', 'Fy', 'Fz'], y_vars=['Sx', 'Sy', 'Sz'], kind="reg", size=5,
                 aspect=0.7)
    plt.show()

    FTsensor = pd.DataFrame(all_data, columns=['Fx', 'Fy', 'Fz'])
    L3sensor = pd.DataFrame(all_data, columns=['Sx', 'Sy', 'Sz'])

    x_train, x_test, y_train, y_test = train_test_split(L3sensor, FTsensor, train_size=0.8, shuffle=False) #random_state=100

    reg = LinearRegression(fit_intercept=False)
    model = reg.fit(x_train, y_train)

    print(model.intercept_)

    print(model.coef_)
    np.save("./forcematrix.npy")

    y_pred = model.predict(x_test)
    print(f"y_pre shape is {y_pred.shape}")

    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_pred))
    # calculate RMSE by hand
    print("RMSE by hand:", sum_erro)
    #
    plt.figure()
    plt.subplot(311)
    plt.plot(range(len(y_pred[:, 0])), np.array(y_test)[:, 0], 'r', label="FTsensor")
    plt.plot(range(len(y_pred[:,0])), y_pred[:,0], 'b', label="L3sensor")
    plt.legend(loc="upper right")

    plt.subplot(312)
    plt.plot(range(len(y_pred[:, 0])), np.array(y_test)[:, 1], 'r', label="FTsensor")
    plt.plot(range(len(y_pred[:, 0])), y_pred[:, 1], 'b', label="L3sensor")
    plt.legend(loc="upper right")

    plt.subplot(313)
    plt.plot(range(len(y_pred[:, 0])), np.array(y_test)[:, 2], 'r', label="FTsensor")
    plt.plot(range(len(y_pred[:, 0])), y_pred[:, 2], 'b', label="L3sensor")
    plt.legend(loc="upper right")

    plt.xlabel("Sample Data")
    plt.ylabel('Force [N]')
    plt.show()





