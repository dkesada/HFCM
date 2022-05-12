

class HFCM():
    def __init__(self, args):
        self._weights = None
        self._input_weights = None
        self._transform_foo = None
        self._window_size = None

    def train_weights(self):
        self._weights = None
        self._input_weights = None

    def forecast(self, dt, len):
        return 0
