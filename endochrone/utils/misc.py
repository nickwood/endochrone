# -*- coding: utf-8 -*-
import cProfile
import pstats
from pstats import SortKey

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


@run_once
def lazy_test_runner(filename=None, verbose=False, printstdout=True):
    """Runs a pytest session with the given filename, if not provided it will
    default to the the currently executing __main__ file. This allows us to use
    the right-click -> 'run in terminal' feature of vscode to execute a single
    test file. This also ensures that built-in fixtures are loaded correctly"""
    import os
    import pytest
    import __main__

    if filename is None:
        filename = os.path.abspath(__main__.__file__)
    args = [filename[filename.find('tests'):]]
    if verbose:
        args.append('-v')
    if printstdout:
        args.append('-s')
    pytest.main(args)


class EndoProfiler():
    def __init__(self):
        pass

    def __enter__(self):
        self.pr = cProfile.Profile()
        self.pr.enable()
        return self

    def __exit__(self, *args):
        self.pr.disable()
        self.ps = pstats.Stats(self.pr).sort_stats(SortKey.CUMULATIVE)
        self.ps.print_stats('python.endochrone', 20)
