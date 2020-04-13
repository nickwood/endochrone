def lazy_test_runner(filename, verbose=False):
    """Runs a pytest session with the given filename. This allows us to use the
    right-click -> 'run in terminal' feature of vscode for a single test file.
    This also ewnsures that built-in fixtures are loaded correctly"""
    import pytest
    args = ['tests/' + filename]
    if verbose:
        args.append('-v')
    pytest.main(args)
