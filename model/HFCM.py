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
                 max_iter=100, perform_idx=1e-05, window_size=4, amount=4, exp_name=None, save_path='output'):
        self._step = self._step_switch(step)
        self._transform_foo = self._trans_switch(transform_foo)
        self._error = self._error_switch(error)
        self._mode = self._mode_switch(mode)
        self._optim = optim
        self._max_iter = max_iter
        self._perform_idx = perform_idx
        self._window_size = window_size
        self._amount = amount  # Amount of training datasets in the original. Maybe number of folds in this one
        self._exp_name = exp_name
        self._save_path = save_path
        self._n_fuzzy_nodes = None
        self._weights = None
        self._input_weights = None
        self._var_names = None
        self._errors = None
        self._loop_error = None
        self._max_vals = None
        self._min_vals = None

    def train_weights(self, dt_train, save=True):
        self._n_fuzzy_nodes = dt_train.shape[1]  # As many fuzzy nodes as variables in our data
        self._input_weights = np.random.rand(self._window_size, self._n_fuzzy_nodes)
        self._weights = np.random.rand(self._n_fuzzy_nodes, self._n_fuzzy_nodes)
        self._errors = []
        self._var_names = list(dt_train.columns)
        dt_train = self._max_min_norm(dt_train)

        t0 = time.time()

        for _ in trange(self._max_iter, desc='model iterations', leave=True):
            self._weights, self._input_weights, self._loop_error = self._mode(
                np.array(dt_train),  # The original trains the model choosing a different train subset in each loop
                self._n_fuzzy_nodes, self._window_size,
                self._step, self._transform_foo(),
                self._weights, self._input_weights,
                self._error, self._optim)

            print('loop_error: ', self._loop_error)

            self._errors.append(self._loop_error)

            if self._loop_error <= self._perform_idx:
                break
        print('Elapsed training time: ', time.time() - t0)

        if save:
            self.save_model()

    # This just forecasts, nothing else. Experimentation should be outside
    def forecast(self, dt, length, obj_vars, print_res=True, plot_res=True):
        if not isinstance(obj_vars, list):
            raise TypeError("The 'obj_vars' argument has to be a list.")
        series = np.array(self._max_min_norm(dt))
        test_errors = {'mae': [0] * len(obj_vars), 'mape': [0] * len(obj_vars)}  # I'll tailor the forecast to use MAE and MAPE
        idx_vars = self._find_idx(obj_vars)
        pred_ts = np.zeros((length, len(obj_vars)))
        orig_ts = np.matrix(dt.iloc[self._window_size:(self._window_size + length), idx_vars])  # It isn't always available in real world applications

        x_window = np.copy(series[0:self._window_size])  # I'll create and move the window myself, without the step

        t0 = time.time()

        for i in range(length):
            pred = modes.calc(self._transform_foo(), self._weights, self._input_weights, x_window)
            pred_ts[i, :] = [pred[j] for j in idx_vars]
            x_window = self._move_window(x_window, pred)

        t1 = time.time() - t0

        test_errors['mae'] = self._calc_real_error(orig_ts, pred_ts, idx_vars, err.mae)
        test_errors['mape'] = self._calc_real_error(orig_ts, pred_ts, idx_vars, err.mape)

        if print_res:
            print('Forecasting results:')
            print(f'MAE: {test_errors["mae"]:.3f}; MAPE: {test_errors["mape"]:.3f}; Execution time: {t1:.3f}')

        if plot_res:
            self._plot_res(series[0:self._window_size], orig_ts, pred_ts, idx_vars)

        return pred_ts, test_errors

    def _move_window(self, x_window, pred):
        x_window[0:(self._window_size-1)] = x_window[1:self._window_size]
        x_window[self._window_size-1] = pred

        return x_window

    def _calc_real_error(self, orig, pred, idx_vars, err_foo):
        un_pred = self._undo_max_min_norm(pred, idx_vars)
        res = None

        if orig.shape[1] == 1:
            res = err_foo(orig, un_pred)
        else:
            res = [err_foo(orig[i], un_pred[i]) for i in idx_vars]

        return res

    def summarize(self):
        res = {
            'config': {
                'step': self._step.__name__,
                'algorithm': self._optim,
                'error': self._error.__name__,
                'transformation function': self._transform_foo.__name__,
                'calculations position': self._mode.__name__,
                'max iterations': self._max_iter,
                'window size': self._window_size,
                'performance index': self._perform_idx,
                'amount': self._amount,
                'save path': self._save_path
            },
            'files': {
                'variable names': self._var_names,
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
                      window_size=summary['config']['window size'], amount=summary['config']['amount'],
                      exp_name=summary['files']['experiment'], save_path=summary['config']['save path'])
        self._weights = np.array(summary['weights']['fcm'])
        self._input_weights = np.array(summary['weights']['aggregation'])
        self._var_names = summary['files']['variable names']
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

    def _find_idx(self, obj_vars):
        return [self._var_names.index(i) for i in obj_vars]

    def _plot_res(self, ini_point, orig_ts, pred_ts, idx_vars):
        ini_point = self._undo_max_min_norm(ini_point, idx_vars)
        pred_ts = self._undo_max_min_norm(pred_ts, idx_vars)
        if orig_ts.shape[1] == 1:
            self._plot_pred(ini_point[:, idx_vars[0]], orig_ts, pred_ts, self._var_names[idx_vars[0]])
        for i in idx_vars:
            self._plot_pred(ini_point[:, i], orig_ts[:, i], pred_ts[:, i], self._var_names[i])

    def _plot_pred(self, ini_point, orig_ts, pred_ts, var_name):
        plt.figure()
        plt.plot(ini_point)
        plt.plot(orig_ts)
        plt.plot(pred_ts)
        plt.xlabel("Time")
        plt.ylabel(var_name)
        plt.show()
