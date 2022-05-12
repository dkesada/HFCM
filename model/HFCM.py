import json
import os
import random
import sys
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

from util import modes, errors as err, data, transformations as trans, steps


class HFCM:
    def __init__(self, step, transform_foo, error, mode, max_iter, perform_idx, window_size, amount, save_path):
        self._step = self._step_switch(step)
        self._transform_foo = self._trans_switch(transform_foo)
        self._error = self._error_switch(error)
        self._mode = self._mode_switch(mode)
        self._max_iter = max_iter
        self._perform_idx = perform_idx
        self._window_size = window_size
        self._amount = amount
        self._save_path = save_path
        self._n_fuzzy_nodes = None
        self._weights = None
        self._input_weights = None
        self._max_vals = None
        self._min_vals = None

    def train_weights(self, dt_train):
        self._weights = None
        self._input_weights = None

    def forecast(self, dt, len):
        return 0

    def save_model(self):
        return 0

    def load_model(self):
        return 0

    # I'll code the original argument switches as static methods of the class
    @staticmethod
    def _step_switch(step):
        res = steps.distinct_steps
        if step == "overlap":
            res = steps.overlap_steps

        return res

    @staticmethod
    def _trans_switch(transform):
        res = trans.gaussian
        if transform == "sigmoid":
            res = trans.sigmoid
        elif transform == "binary":
            res = trans.binary
        elif transform == "tanh":
            res = trans.tanh
        elif transform == "arctan":
            res = trans.arctan

        return res

    @staticmethod
    def _error_switch(error):
        res = err.max_pe
        if error == "rmse":
            res = err.rmse
        elif error == "mpe":
            res = err.mpe

        return res

    @staticmethod
    def _mode_switch(mode):
        res = modes.inner_calculations
        if mode == "outer":
            res = modes.outer_calculations

        return res

    # I'll do the max/min normalization of the dataset via pandas, not with a lambda
    def _max_min_norm(self, dt):
        self._min_vals = dt.min()
        self._max_vals = dt.max()

        return (dt - self._min_vals) / (self._max_vals - self._min_vals)

