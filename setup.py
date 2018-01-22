#!/usr/bin/env python
#
# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup

import versioneer


DISTNAME = "empyrical"
DESCRIPTION = """empyrical is a Python library with performance and risk \
statistics commonly used in quantitative finance"""
LONG_DESCRIPTION = """empyrical is a Python library with performance and risk
statistics commonly used in quantitative finance by `Quantopian Inc`_.

.. _Quantopian Inc: https://www.quantopian.com
.. _Zipline: http://zipline.io
.. _pyfolio: http://quantopian.github.io/pyfolio/
"""
MAINTAINER = "Quantopian Inc"
MAINTAINER_EMAIL = "opensource@quantopian.com"
AUTHOR = "Quantopian Inc"
AUTHOR_EMAIL = "opensource@quantopian.com"
URL = "https://github.com/quantopian/empyrical"
LICENSE = "Apache License, Version 2.0"

classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.5",
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Operating System :: OS Independent"
]


test_reqs = [
    "nose>=1.3.7",
    "parameterized>=0.6.1"
]


requirements = [
    'numpy>=1.9.2',
    'pandas>=0.16.1',
    'scipy>=0.15.1',
    "pandas-datareader>=0.2",
    'bottleneck>=1.0.0'
]


extras_requirements = {
    "dev": [
        "nose==1.3.7",
        "parameterized==0.6.1",
        "flake8==2.5.1"
    ]
}


if __name__ == "__main__":
    setup(
        name=DISTNAME,
        cmdclass=versioneer.get_cmdclass(),
        version=versioneer.get_version(),
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        long_description=LONG_DESCRIPTION,
        packages=["empyrical", "empyrical.tests"],
        classifiers=classifiers,
        install_requires=requirements,
        extras_require=extras_requirements,
        tests_require=test_reqs,
        test_suite="nose.collector"
    )
