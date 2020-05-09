import itertools
import numpy as np
from typing import Dict, Iterable


class GridSearch:
    def __init__(self, *, precision=14, find_minimum=True, find_maximum=True):
        self.precision = precision
        self.find_minimum = find_minimum
        self.find_maximum = find_maximum

    def fit(self, search_space: Dict[str, Iterable], scoring_func: callable):
        '''given:
        def function(*, a, b):
            return 1.75 + (a - 0.5) ** 2 + (b - 0.75)
        search_space = {'a': np.arange(0.0, 3.0, 0.05),
                        'b': np.arange(0, 3, 0.05)}

        gs = GridSearch()
        gs.fit(search_space, function)
        gs.minimum -> 1.75
        gs.min_args -> {'a': 0.5, 'b': 0.75}'''

        searches = itertools.product(*search_space.values())
        keys = search_space.keys()
        minimum, maximum = None, None
        min_args, max_args = {}, {}

        for s in searches:
            s = tuple(np.round(s, decimals=self.precision))
            f_args = dict(zip(keys, s))
            val = scoring_func(**f_args)
            if self.find_minimum and (minimum is None or val < minimum):
                minimum = round(val, self.precision)
                min_args = f_args
            if self.find_maximum and (maximum is None or val > maximum):
                maximum = round(val, self.precision)
                max_args = f_args

        self.minimum = minimum
        self.min_args = min_args

        self.maximum = maximum
        self.max_args = max_args
        return self
