from model.HFCM import HFCM
import pandas as pd

if __name__ == '__main__':
    model = HFCM(optim='Nelder-Mead')
    # Prepro: normalization and cv
    dt = pd.read_csv("data/dt_cycles.csv")
    dt = dt.iloc[0:200, :]
    del dt['cyc']
    model.train_weights(dt)
    #model.load_model("HFCM_Tue_May_17_17_42_04_2022.json")
    model.summarize()
