========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions| |requires|
        | |codecov|
    * - package
      - | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/nb_functools/badge/?style=flat
    :target: https://nb_functools.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/bfulroth/nb_functools/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/bfulroth/nb_functools/actions

.. |requires| image:: https://requires.io/github/bfulroth/nb_functools/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/bfulroth/nb_functools/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/bfulroth/nb_functools/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/bfulroth/nb_functools

.. |commits-since| image:: https://img.shields.io/github/commits-since/bfulroth/nb_functools/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/bfulroth/nb_functools/compare/v0.0.0...master



.. end-badges

A package containing useful functions for data wrangling and analysis in jupyter notebook

* Free software: MIT license

Installation
============

::

    pip install nb-functools

You can also install the in-development version with::

    pip install https://github.com/bfulroth/nb_functools/archive/master.zip


Documentation
=============


https://nb_functools.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
