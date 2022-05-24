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
    cte_cols = ['rho_1', 'C_p1', 'C_in', 'vol', 'C_ain']
    # model.train_weights(dt, idx_var, cv_size=4, cte_cols=['rho_1', 'C_p1', 'C_in', 'vol', 'C_ain'])
    model.load_model("HFCM_Mon_May_23_15_57_35_2022.json")
    # model.forecast(dt_test.iloc[2:40], length=4, obj_vars=["T_1", "T_2"])
    model.forecast(dt_test.iloc[2:240], length=80, obj_vars=["T_1"])
    model.summarize()

