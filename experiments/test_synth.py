from model.HFCM import HFCM
from util.cv_controller import CvCtrl
import pandas as pd
from random import seed

if __name__ == '__main__':
    model_init = HFCM

    # Config
    config = {
        'optim': 'Nelder-Mead',
        'window_size': 4,
        'max_iter_optim': 1e5,
        'perform_idx': 1e-7,
        'max_iter': 100,
        'cte_cols': ['rho_1', 'C_p1', 'C_in', 'vol', 'C_ain']
    }


    dt_file = "dt_synth_unfolded.csv"
    info_file = "exec_info_synth.txt"
    output_file = "HFCM_mae_mape_synth.csv"
    seed = 42
    pred_len = 96

    pd.options.mode.chained_assignment = None  # I've had enough of those bloody false positive assignment on copy warnings
    ctrl = CvCtrl(dt_file, info_file, seed)
    ctrl.run(model_init, pred_len, output_file, **config)


