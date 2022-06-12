from model.HFCM import HFCM
from util.cv_controller import CvCtrl
import pandas as pd

if __name__ == '__main__':
    model_init = HFCM

    # Config
    config = {
        'optim': 'Nelder-Mead',
        'transform_foo': 'sigmoid',
        'window_size': 2,
        'diff': True,
        'max_iter_optim': 1e4,
        'perform_idx': 1e-7,
        #'mode': 'inner',
        'amount': 2,
        'max_iter': 20
    }

    dt_file = "dt_motor_red_unfolded.csv"
    info_file = "exec_info_motor.txt"
    output_file = "HFCM_mae_mape_motor.csv"
    seed = 42
    pred_len = 20

    pd.options.mode.chained_assignment = None  # I've had enough of those bloody false positive assignment on copy warnings
    ctrl = CvCtrl(dt_file, info_file, seed)
    ctrl.run(model_init, pred_len, output_file, **config)
