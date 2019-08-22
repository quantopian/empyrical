#! /usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import warnings


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        loader = unittest.TestLoader()
        tests = loader.discover('.')
        testRunner = unittest.runner.TextTestRunner()
        testRunner.run(tests)
