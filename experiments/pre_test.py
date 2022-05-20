from model.HFCM import HFCM
import pandas as pd

if __name__ == '__main__':
    model = HFCM(optim='Nelder-Mead')
    # Prepro: normalization and cv
    dt = pd.read_csv("data/dt_cycles.csv")
    dt_test = dt.iloc[1100:1199, :]
    dt = dt.iloc[0:1000, :]  # 0:200 con 10000 its en el optim
    del dt['cyc']
    del dt_test['cyc']
    # model.train_weights(dt)
    model.load_model("HFCM_Wed_May_18_18_27_11_2022.json")
    #model.forecast(dt_test, length=4, obj_vars=["T_1", "T_2"])
    model.forecast(dt_test, length=4, obj_vars=["T_1"])
    model.summarize()

