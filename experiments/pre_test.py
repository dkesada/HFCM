from model.HFCM import HFCM
import pandas as pd
from random import seed

if __name__ == '__main__':
    model = HFCM(optim='Nelder-Mead', window_size=4, max_iter_optim=1e4, perform_idx=1e-7, max_iter=300)
    # Prepro: normalization and cv
    dt = pd.read_csv("data/dt_synth_unfolded.csv")
    idx_var = 'cyc'
    dt_test = dt.iloc[4100:5199, :]
    dt = dt.iloc[0:3999, :]
    # del dt[idx_var]
    del dt_test[idx_var]
    cte_cols = ['rho_1', 'C_p1', 'C_in', 'vol', 'C_ain']
    seed(42)
    # model.train_weights(dt, idx_var, cv_size=4, cte_cols=cte_cols)
    model.load_model("HFCM_Thu_May_26_00_45_33_2022.json")
    # model.forecast(dt_test.iloc[2:40], length=4, obj_vars=["T_1", "T_2"])
    model.forecast(dt_test.iloc[0:99], length=89, obj_vars=["T_1"])
    print(model.summarize())

