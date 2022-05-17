import json
import os
import random
import sys
import time
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange

from util import modes, errors as err, data, transformations as trans, steps


class HFCM:
    def __init__(self, step='overlap', transform_foo='sigmoid', error='rmse', mode='outer',
                 max_iter=100, perform_idx=1e-05, window_size=4, amount=4, exp_name=None, save_path='output'):
        self._step = self._step_switch(step)
        self._transform_foo = self._trans_switch(transform_foo)
        self._error = self._error_switch(error)
        self._mode = self._mode_switch(mode)
        self._max_iter = max_iter
        self._perform_idx = perform_idx
        self._window_size = window_size
        self._amount = amount  # Amount of training datasets in the original. Maybe number of folds in this one
        self._exp_name = exp_name
        self._save_path = save_path
        self._n_fuzzy_nodes = None
        self._weights = None
        self._input_weights = None
        self._errors = None
        self._loop_error = None
        self._max_vals = None
        self._min_vals = None

    def train_weights(self, dt_train, save=True):
        self._n_fuzzy_nodes = dt_train.shape[1]  # As many fuzzy nodes as variables in our data
        self._input_weights = np.random.rand(self._window_size, self._n_fuzzy_nodes)
        self._weights = np.random.rand(self._n_fuzzy_nodes, self._n_fuzzy_nodes)
        self._errors = []
        dt_train = self._max_min_norm(dt_train)

        t0 = time.time()

        # for _ in trange(self._max_iter, desc='model iterations', leave=True):
        #     self._weights, self._input_weights, self._loop_error = self._mode(
        #         np.array(dt_train),  # The original trains the model choosing a different train subset in each loop
        #         self._n_fuzzy_nodes, self._window_size,
        #         self._step, self._transform_foo(),
        #         self._weights, self._input_weights,
        #         self._error)
        #
        #     print('loop_error: ', self._loop_error)
        #
        #     self._errors.append(self._loop_error)
        #
        #     if self._loop_error <= self._perform_idx:
        #         break
        print('Elapsed training time: ', t0 - time.time())

        if save:
            self.save_model()

    def forecast(self, dt, len):
        return 0

    def summarize(self):
        res = {
            'config': {
                'step': self._step.__name__,
                'algorithm': 'Nelder-Mead',
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
        self._min_vals = dt.min()
        self._max_vals = dt.max()

        return (dt - self._min_vals) / (self._max_vals - self._min_vals)

