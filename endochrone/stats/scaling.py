# -*- coding: utf-8 -*-
import numpy as np

from endochrone import Base

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class FeatureScaling(Base):
    def __init__(self, method='z_score'):
        if method not in ['min_max', 'mean_norm', 'z_score']:
            raise ValueError('Unknown method: %s' % method)
        self.method = method
        super().__init__(properties={'requires_targets': False})

    def fit(self, *, features, targets=None):
        if targets is not None:
            self.properties_['requires_targets'] = True
        self.validate_fit(features=features, targets=targets)

        self.feature_scale = {'c_min': np.min(features, axis=0),
                              'c_max': np.max(features, axis=0),
                              'c_mean': np.mean(features, axis=0),
                              'c_std': np.std(features, ddof=1, axis=0)}

        if targets is not None:
            self.targets_scale = {'c_min': np.min(targets, axis=0),
                                  'c_max': np.max(targets, axis=0),
                                  'c_mean': np.mean(targets, axis=0),
                                  'c_std': np.std(targets, ddof=1, axis=0)}

        self.validate_scale()

    def validate_scale(self):
        check_scales = [self.feature_scale]
        if hasattr(self, 'targets_scale'):
            check_scales.append(self.targets_scale)

        for scale in check_scales:
            if self.method in ['min_max', 'mean_norm'] and\
                    np.any(scale['c_max'] - scale['c_min'] == 0):
                raise ValueError("Invalid scale: min == max")
            if self.method == 'z_score' and np.any(scale['c_std'] == 0):
                raise ValueError("Invalid scale: standard deviation == 0")

    def transform(self, *, features=None, targets=None, inverse=False):
        ret = []
        if features is not None:
            f_scale = {**self.feature_scale, 'inverse': inverse}
            ret.append(self.perform_transform(data=features, scale=f_scale))

        if targets is not None:
            if not hasattr(self, 'targets_scale'):
                raise ValueError("Cannot transform targets - no scale fitted")
            t_scale = {**self.targets_scale, 'inverse': inverse}
            ret.append(self.perform_transform(data=targets, scale=t_scale))

        if len(ret) == 1:
            return ret[0]
        return ret

    def perform_transform(self, *, data, scale):
        if self.method == 'min_max':
            s_feat = min_max_scale_(cols=data, **scale)
        elif self.method == 'mean_norm':
            s_feat = mean_norm_scale_(cols=data, **scale)
        else:  # method = 'z_score'
            s_feat = z_score_scale_(cols=data, **scale)
        return s_feat

    def fit_and_transform(self, *, features, targets=None):
        self.fit(features=features, targets=targets)
        return self.transform(features=features, targets=targets)

    def reverse(self, *, features=None, targets=None):
        return self.transform(features=features, targets=targets, inverse=True)


def min_max_scale_(*, cols, c_min, c_max, inverse=False, **kwargs):
    if inverse:
        return cols * (c_max - c_min) + c_min
    return (cols - c_min) / (c_max - c_min)


def mean_norm_scale_(*, cols, c_min, c_max, c_mean, inverse=False, **kwargs):
    if inverse:
        return cols * (c_max - c_min) + c_mean
    return (cols - c_mean) / (c_max - c_min)


def z_score_scale_(*, cols, c_mean, c_std, inverse=False, **kwargs):
    if inverse:
        return cols * c_std + c_mean
    return (cols - c_mean)/c_std
