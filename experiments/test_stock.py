from model.HFCM import HFCM
from util.cv_controller import CvCtrl
import pandas as pd
from random import seed

if __name__ == '__main__':
    model_init = HFCM

    # Config
    config = {
        'optim': 'Nelder-Mead',
        'transform_foo': 'gaussian',
        'window_size': 4,
        'max_iter_optim': 1e6,
        'perform_idx': 1e-7,
        'max_iter': 20
    }

    dt_file = "TWII_1y.csv"
    info_file = "exec_info_stock.txt"
    output_file = "HFCM_mae_mape_stock.csv"
    seed = 42
    pred_len = 1

    pd.options.mode.chained_assignment = None  # I've had enough of those bloody false positive assignment on copy warnings
    ctrl = CvCtrl(dt_file, info_file, seed)
    ctrl.run(model_init, pred_len, output_file, **config)
