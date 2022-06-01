from random import seed
import numpy as np
import pandas as pd
import json


# Specific cross-validation from an info file generated in R
class CvCtrl:
    def __init__(self, dt_file, info_file, seed):
        self._dt = self._load_dt(dt_file)
        self._info = self._load_info(info_file)
        self._idx_var = self._info['idx_cyc']
        self._obj_var = self._info['obj_var']
        self._seed = seed
        self._results = None

    def run(self, model_init, pred_len, output_file=None, **kwargs):
        self._results = [[], [], [], []]
        np.random.seed(seed=self._seed)

        for cv in self._info['cv']:
            cv_res = self._model_eval(model_init, cv, pred_len, **kwargs)
            print("MAE of the fold: ")
            print(np.mean(cv_res[0]))
            print("MAPE of the fold: ")
            print(np.mean(cv_res[1]))
            self._results[0] = self._results[0] + cv_res[0]
            self._results[1] = self._results[1] + cv_res[1]
            self._results[2] = self._results[2] + cv_res[2]
            self._results[3].append(cv_res[3])

        print("Final MAE of the model: ")
        print(np.mean(self._results[0]))
        print("Final MAPE of the model: ")
        print(np.mean(self._results[1]))
        print("Final exec. time of the model: ")
        print(np.mean(self._results[2]))
        print("Final training time of the model: ")
        print(np.mean(self._results[3]))

        if output_file is not None:
            res_data = pd.DataFrame({'MAE': self._results[0], 'MAPE': self._results[1]})
            res_data.to_csv('output/' + output_file)

    # This is model specific of the HFCM, it should be modified for other models
    def _model_eval(self, model_init, cv, pred_len, **kwargs):
        dt_train, dt_test = self._get_train_test(cv)
        cte_cols = kwargs['cte_cols']
        del kwargs['cte_cols']
        model = model_init(**kwargs)
        t_train = model.train_weights(dt_train, self._idx_var, cte_cols=cte_cols, save=False)
        res = [[], [], []]
        pad_size = model._window_size + pred_len  # Number of instances needed for prediction + test

        for idx in dt_test[self._idx_var].unique():
            cyc_test = dt_test[dt_test[self._idx_var] == idx]
            cyc_test = cyc_test.drop(self._idx_var, axis=1)
            for i in range(cyc_test.shape[0] // pad_size):
                _, cyc_errors, cyc_exec_time = model.forecast(cyc_test[i*pad_size:], length=pred_len,
                                                              obj_vars=[self._obj_var], plot_res=False)
                res[0] = res[0] + cyc_errors["mae"]
                res[1] = res[1] + cyc_errors["mape"]
                res[2].append(cyc_exec_time)

        return res + [t_train]

    # From kTsnn utils:

    # Load a dataset stored in the 'data' folder
    @staticmethod
    def _load_dt(file):
        return pd.read_csv("data/" + file)

    # Load the json info file. Its structure is {'obj_var': [...], 'idx_cyc': ..., 'cv': [[...],[...],...]}
    @staticmethod
    def _load_info(file):
        with open("data/" + file) as f:
            info = json.load(f)
        return info

    # Map function that returns a list with the result
    @staticmethod
    def _map_w(func, col):
        return list(map(func, col))

    # Return the x and y dataframes for both train and test
    # The dataframe is divided by cycles, and each cycle is part of a cross-validation fold
    def _get_train_test(self, cv):
        test = cv['test']
        dt_test = self._dt[self._map_w(lambda x: x in test, self._dt[self._idx_var])]
        dt_train = self._dt[self._map_w(lambda x: x not in test, self._dt[self._idx_var])]

        return dt_train, dt_test
