from model.HFCM import HFCM
import pandas as pd

if __name__ == '__main__':
    model = HFCM()
    dt = pd.read_csv("data/dt_cycles.csv")
    dt = dt.iloc[0:7000, :]
    model.train_weights(dt)
    print(model.summarize())
