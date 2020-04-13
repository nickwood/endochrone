def lazy_test_runner(filename=None, verbose=False):
    """Runs a pytest session with the given filename, if not provided it will
    default to the the currently executing __main__ file. This allows us to use
    the right-click -> 'run in terminal' feature of vscode to execute a single
    test file. This also ensures that built-in fixtures are loaded correctly"""
    import os
    import pytest
    import __main__

    if filename is None:
        filename = os.path.basename(__main__.__file__)
    args = ['tests/' + filename]
    if verbose:
        args.append('-v')
    pytest.main(args)
