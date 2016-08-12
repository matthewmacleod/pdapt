pdapt
==============================

pdapt is a Python 3 machine learning suite coded from scratch.

The main purpose is to implement simple algorithms in order to
*prototype new models and ideas*.

pdapt is *not* intended to be a massively distributed real-time solution.


Installation
-----------------

see file:

  Install.md

after install, activate like so:

        . venv/bin/activate


to deactivate environment:

        deactivate


Running
-----------------

    python main.py


Testing
-----------------

To run unit tests:

    ./run_pdapt_tests unit


To run doctests:

    ./run_pdapt_tests doctest


To check doctests for failures:

    ./run_pdapt_tests doctestf



Documentation
---------------

      cd doc

      make html

      firefox _build/html/index.html


License
---------------

see license.txt file for this



