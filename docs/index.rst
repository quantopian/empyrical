empyrical
=========

Common financial risk metrics.

Table of Contents
-----------------

-  `Installation <#installation>`__
-  `Usage <#usage>`__
-  `API <#api>`__
-  `Support <#support>`__
-  `Contributing <#contributing>`__
-  `Testing <#testing>`__
-  `Documentation <#documentation>`__

Source location
---------------
Hosted on GitHub: https://github.com/quantopian/empyrical

Installation
------------

::

    pip install empyrical

Usage
-----

Simple Statistics

.. code:: python

    import numpy as np
    from empyrical import max_drawdown, alpha_beta

    returns = np.array([.01, .02, .03, -.4, -.06, -.02])
    benchmark_returns = np.array([.02, .02, .03, -.35, -.05, -.01])

    # calculate the max drawdown
    max_drawdown(returns)

    # calculate alpha and beta
    alpha, beta = alpha_beta(returns, benchmark_returns)

Rolling Measures

.. code:: python

    import numpy as np
    from empyrical import roll_max_drawdown

    returns = np.array([.01, .02, .03, -.4, -.06, -.02])

    # calculate the rolling max drawdown
    roll_max_drawdown(returns, window=3)

Pandas Support

.. code:: python

    import pandas as pd
    from empyrical import roll_up_capture, capture

    returns = pd.Series([.01, .02, .03, -.4, -.06, -.02])

    # calculate a capture ratio
    capture(returns)

    # calculate capture for up markets on a rolling 60 day basis
    roll_up_capture(returns, window=60)

API
---

.. toctree::
   :maxdepth: 2

   modules.rst

Support
-------

Please `open an
issue <https://github.com/quantopian/empyrical/issues/new>`__ for
support.

Contributing
------------

Please contribute using `Github
Flow <https://guides.github.com/introduction/flow/>`__. Create a branch,
add commits, and `open a pull
request <https://github.com/quantopian/empyrical/compare/>`__.

Testing
-------

-  install requirements
-  "nose>=1.3.7",
-  "parameterized>=0.6.1"

::

    python -m unittest


Documentation
-------------
To build this documentation requires `sphinx`

::

    cd docs
    make_docs.bat


.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
