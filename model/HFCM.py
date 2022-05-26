import json
import os
import random
import sys
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange

from util import modes, errors as err, transformations as trans, steps


class HFCM:
    def __init__(self, step='overlap', transform_foo='sigmoid', error='rmse', mode='outer', optim='Nelder-Mead',
                 max_iter=100, max_iter_optim=1e4, perform_idx=1e-05, window_size=4, amount=4, exp_name=None,
                 save_path='output'):
        self._step = self._step_switch(step)
        self._transform_foo = self._trans_switch(transform_foo)
        self._error = self._error_switch(error)
        self._mode = self._mode_switch(mode)
        self._optim = optim
        self._max_iter = max_iter
        self._max_iter_optim = max_iter_optim
        self._perform_idx = perform_idx
        self._window_size = window_size
        self._amount = amount  # Amount of training datasets in the original. Maybe number of folds in this one
        self._exp_name = exp_name
        self._save_path = save_path
        self._n_fuzzy_nodes = None
        self._weights = None
        self._input_weights = None
        self._var_names = None
        self._cte_cols = None
        self._idx_var = None
        self._errors = None
        self._loop_error = None
        self._max_vals = None
        self._min_vals = None

    def train_weights(self, dt_train, idx_var=None, cv_size=1, cte_cols=None, save=True):
        unique_cyc = dt_train[idx_var].unique()
        dt_train = self._diff_ts(dt_train, idx_var, cte_cols)  # Differentiate the data to remove the tendency of the ts
        tmp_idx_var = dt_train.pop(idx_var)
        self._n_fuzzy_nodes = dt_train.shape[1] # As many fuzzy nodes as variables in our data
        self._input_weights = np.random.rand(self._window_size, self._n_fuzzy_nodes)
        self._weights = np.random.rand(self._n_fuzzy_nodes, self._n_fuzzy_nodes)
        self._errors = []
        self._var_names = list(dt_train.columns)
        self._cte_cols = cte_cols
        self._idx_var = idx_var
        dt_train = self._max_min_norm(dt_train)  # Normalization of the data
        dt_train.loc[:, idx_var] = tmp_idx_var
        del tmp_idx_var

        t0 = time.time()

        for _ in trange(self._max_iter, desc='model iterations', leave=True):
            self._weights, self._input_weights, self._loop_error = self._mode(
                self._get_random_cycles(dt_train, cv_size, idx_var, unique_cyc),  # The original trains the model choosing a different train subset in each loop
                self._n_fuzzy_nodes, self._window_size,
                self._step, self._transform_foo(),
                self._weights, self._input_weights,
                self._error, self._optim, self._max_iter_optim)

            print('loop_error: ', self._loop_error)

            self._errors.append(self._loop_error)

            if self._loop_error <= self._perform_idx:
                break
        t1 = time.time() - t0
        print('Elapsed training time: ', t1)

        if save:
            self.save_model()

        return t1

    # This just forecasts, nothing else. Experimentation should be outside
    def forecast(self, dt, length, obj_vars, print_res=True, plot_res=True):
        if not isinstance(obj_vars, list):
            raise TypeError("The 'obj_vars' argument has to be a list.")
        ini_vals = dt.iloc[self._window_size-1]  # First values needed to undo the differentiation
        series = np.array(self._max_min_norm(self._diff_ts(dt, self._idx_var, self._cte_cols)))
        test_errors = {'mae': [0] * len(obj_vars), 'mape': [0] * len(obj_vars)}  # I'll tailor the forecast to use MAE and MAPE
        idx_obj_vars = self._find_idx(obj_vars)
        pred_ts = np.zeros((length, len(obj_vars)))
        orig_ts = np.array(dt.iloc[self._window_size:(self._window_size + length), idx_obj_vars])  # It isn't always available in real world applications

        x_window = np.copy(series[0:self._window_size])  # I'll create and move the window myself, without the step

        t0 = time.time()

        for i in range(length):
            pred = modes.calc(self._transform_foo(), self._weights, self._input_weights, x_window)
            pred_ts[i, :] = [pred[j] for j in idx_obj_vars]
            x_window = self._move_window(x_window, pred)

        t1 = time.time() - t0

        pred_ts = self._undo_diff(self._undo_max_min_norm(pred_ts, idx_obj_vars), ini_vals, idx_obj_vars)

        test_errors['mae'] = self._calc_real_error(orig_ts, pred_ts, idx_obj_vars, err.mae)
        test_errors['mape'] = self._calc_real_error(orig_ts, pred_ts, idx_obj_vars, err.mape)

        if print_res:
            print('Forecasting results:')
            for i in range(len(obj_vars)):
                print(f'Results for {obj_vars[i]}:')
                print(f'MAE: {test_errors["mae"][i]:.3f}; MAPE: {test_errors["mape"][i]:.3f}')
            print(f'Execution time: {t1:.3f}')

        if plot_res:
            self._plot_res(np.array(dt.iloc[0:self._window_size]), orig_ts, pred_ts, idx_obj_vars)

        return pred_ts, test_errors, t1

    def _move_window(self, x_window, pred):
        x_window[0:(self._window_size-1)] = x_window[1:self._window_size]
        x_window[self._window_size-1] = pred

        return x_window

    def _calc_real_error(self, orig, pred, idx_vars, err_foo):
        return [err_foo(orig[:, i], pred[:, i]) for i in range(len(idx_vars))]

    def summarize(self):
        res = {
            'config': {
                'step': self._step.__name__,
                'algorithm': self._optim,
                'error': self._error.__name__,
                'transformation function': self._transform_foo.__name__,
                'calculations position': self._mode.__name__,
                'max iterations': self._max_iter,
                'max iterations optim': self._max_iter_optim,
                'window size': self._window_size,
                'performance index': self._perform_idx,
                'amount': self._amount,
                'save path': self._save_path
            },
            'files': {
                'variable names': self._var_names,
                'constant columns': self._cte_cols,
                'index variable': self._idx_var,
                'experiment': self._exp_name
            },
            'weights': {
                'aggregation': self._input_weights.tolist(),
                'fcm': self._weights.tolist(),
                'max_vals': self._max_vals.to_json(),
                'min_vals': self._min_vals.to_json()
            },
            'results': {
                'final error': self._loop_error,
                'iterations': len(self._errors),
                'errors': self._errors
            }
        }

        return res

    def save_model(self):
        if self._weights is None or self._input_weights is None:
            raise AttributeError('The model cannot be saved because no weights have been learned yet.')
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)
        file_name = 'HFCM_' + time.asctime().replace(' ', '_').replace(':', '_')
        with open(f'{self._save_path}/{file_name}.json', 'w') as f:
            json.dump(self.summarize(), f)

    def load_model(self, file):
        with open(f"output/{file}", "r") as f:
            summary = json.load(f)

        self.__init__(step=summary['config']['step'], transform_foo=summary['config']['transformation function'],
                      error=summary['config']['error'], mode=summary['config']['calculations position'],
                      max_iter=summary['config']['max iterations'], perform_idx=summary['config']['performance index'],
                      max_iter_optim=summary['config']['max iterations optim'],
                      window_size=summary['config']['window size'], amount=summary['config']['amount'],
                      exp_name=summary['files']['experiment'], save_path=summary['config']['save path'])
        self._weights = np.array(summary['weights']['fcm'])
        self._input_weights = np.array(summary['weights']['aggregation'])
        self._var_names = summary['files']['variable names']
        self._cte_cols = summary['files']['constant columns']
        self._idx_var = summary['files']['index variable']
        self._min_vals = pd.Series(json.loads(summary['weights']['min_vals']))
        self._max_vals = pd.Series(json.loads(summary['weights']['max_vals']))
        self._loop_error = summary['results']['final error']
        self._errors = summary['results']['errors']


    # I'll code the original argument switches as static methods of the class
    @staticmethod
    def _step_switch(step):
        res = steps.distinct_steps
        if step == 'overlap':
            res = steps.overlap_steps

        return res

    @staticmethod
    def _trans_switch(transform):
        res = trans.gaussian
        if transform == 'sigmoid':
            res = trans.sigmoid
        elif transform == 'binary':
            res = trans.binary
        elif transform == 'tanh':
            res = trans.tanh
        elif transform == 'arctan':
            res = trans.arctan

        return res

    @staticmethod
    def _error_switch(error):
        res = err.max_pe
        if error == 'rmse':
            res = err.rmse
        elif error == 'mpe':
            res = err.mpe

        return res

    @staticmethod
    def _mode_switch(mode):
        res = modes.inner_calculations
        if mode == 'outer':
            res = modes.outer_calculations

        return res

    # I'll do the max/min normalization of the dataset via pandas, not with a lambda
    def _max_min_norm(self, dt):
        if self._min_vals is None or self._max_vals is None:
            self._min_vals = dt.min()
            self._max_vals = dt.max()

        return (dt - self._min_vals) / (self._max_vals - self._min_vals)

    def _get_max_vals(self, idx_vars):
        return np.array(self._max_vals.iloc[idx_vars])

    def _get_min_vals(self, idx_vars):
        return np.array(self._min_vals.iloc[idx_vars])

    def _undo_max_min_norm(self, pred, idx_vars):
        return pred * (self._get_max_vals(idx_vars) - self._get_min_vals(idx_vars)) + self._get_min_vals(idx_vars)

    def _diff_ts(self, dt, idx_var, cte_cols):
        if cte_cols is None and idx_var in dt.columns:
            excluded_cols = dt[idx_var]
        elif cte_cols is not None:
            if idx_var in dt.columns:
                cte_cols = cte_cols + [idx_var]
            excluded_cols = dt[cte_cols]

        if idx_var in dt.columns:
            for i in dt[idx_var].unique():  # Differentiate at cycle level to not mix them
                dt.loc[dt[idx_var] == i, :] = dt[dt[idx_var] == i].diff()
        else:
            dt = dt.diff()
        if cte_cols is not None:
            dt.loc[:, cte_cols] = excluded_cols

        return dt.dropna(axis=0)

    def _undo_diff(self, ts, ini_vals, idx_vars):
        ts[0] = ts[0] + ini_vals[idx_vars]
        for i in range(1, len(ts)):
            ts[i] = ts[i-1] + ts[i]

        return ts

    @staticmethod
    def _get_random_cycles(dt, n, idx_var, unique_cyc):
        cycles = np.random.choice(unique_cyc, n, replace=False)
        return np.array(dt[dt[idx_var].isin(cycles)].drop(idx_var, axis=1))

    def _find_idx(self, obj_vars):
        return [self._var_names.index(i) for i in obj_vars]

    def _plot_res(self, ini_point, orig_ts, pred_ts, idx_vars):
        for i in range(len(idx_vars)):
            self._plot_pred(ini_point[:, idx_vars[i]], orig_ts[:, i], pred_ts[:, i], idx_vars[i])

    def _plot_pred(self, ini_point, orig_ts, pred_ts, var_idx):
        ini_point = np.concatenate((ini_point, [None]*self._window_size))  # I pad the series with None for plotting purposes
        switch_point = [ini_point[self._window_size-1]]
        orig_ts = np.concatenate(([None]*(self._window_size-1), switch_point, orig_ts))
        pred_ts = np.concatenate(([None]*(self._window_size-1), switch_point, pred_ts))

        plt.figure()
        plt.plot(ini_point)
        plt.plot(orig_ts)
        plt.plot(pred_ts)
        plt.xlabel("Time")
        plt.ylabel(self._var_names[var_idx])
        plt.show()

