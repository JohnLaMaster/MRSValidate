.. _installation:

##############
Installation
##############

Requirements
=============

mrsvalidate requires **Python 3.11+** and the following scientific
dependencies, installed automatically:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Package
     - Minimum version
     - Role
   * - `NumPy <https://numpy.org>`_
     - 1.26
     - Arrays, vectorized arithmetic, linear algebra primitives.
   * - `SciPy <https://scipy.org>`_
     - 1.11
     - Statistical distributions, resampling, Theil-Sen, permutation
       tests.
   * - `Statsmodels <https://www.statsmodels.org>`_
     - 0.14
     - Mixed-effects (REML) fitting, cluster-robust covariance,
       Lilliefors critical values.

Two optional extras add further capability:

- **viz** — `Matplotlib <https://matplotlib.org>`_, required for
  :mod:`mrsvalidate.viz` plotting functions.
- **pandas** — `pandas <https://pandas.pydata.org>`_, used only as an
  adapter for CSV loading/export, never as the core numerical
  representation.

From source (current development version)
=============================================

.. code-block:: console

    $ git clone https://github.com/JohnLaMaster/MRSValidate.git
    $ cd MRSValidate
    $ pip install -e ".[dev]"

The ``dev`` extra additionally installs ``pytest``, ``hypothesis``, and
``build``, and pulls in the ``viz``/``pandas`` extras — everything
needed to run the test suite and build a release locally.

With only the plotting/export extras
=======================================

.. code-block:: console

    $ pip install "mrsvalidate[viz,pandas]"

Core-only (no plotting, no DataFrame adapters)
=================================================

.. code-block:: console

    $ pip install mrsvalidate

Verifying the installation
=============================

.. code-block:: python

    >>> import mrsvalidate
    >>> mrsvalidate.__version__  # doctest: +SKIP
    '0.1.0.dev0'

Building the documentation locally
=====================================

.. code-block:: console

    $ pip install -r requirements-docs.txt
    $ cd docs
    $ make html
    $ open build/html/index.html  # macOS; use `xdg-open` on Linux

License
=========

The project license has not yet been finalized (pending the
third-party-code provenance review described in ``THIRD_PARTY_CODE.md``
in the repository root). No installation step requires accepting a
license beyond those of the dependencies listed above.
