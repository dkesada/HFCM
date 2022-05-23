from model.HFCM import HFCM
import pandas as pd

if __name__ == '__main__':
    model = HFCM(optim='Nelder-Mead', window_size=4)
    # Prepro: normalization and cv
    dt = pd.read_csv("data/dt_cycles.csv")
    idx_var = 'cyc'
    dt_test = dt.iloc[4100:4199, :]
    dt = dt.iloc[0:3999, :]  # 0:200 con 10000 its en el optim
    # del dt[idx_var]
    del dt_test[idx_var]
    model.train_weights(dt, idx_var, cv_size=4)
    # model.load_model("HFCM_Fri_May_20_16_34_42_2022.json")
    # model.forecast(dt_test, length=4, obj_vars=["T_1", "T_2"])
    # model.forecast(dt_test, length=10, obj_vars=["T_1"])
    model.summarize()

