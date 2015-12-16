pdapt
==============================

pdpat is a Python 3 machine learning suite coded from scratch.

The main purpose is to implement simple algorithms in order to
*prototype new models and ideas*.

pdapt is *not* intended to be a massively distributed real-time solution.


Installation
-----------------

see file:

  Install.md

after install, activate like so:

        . bin/activate


to deactivate environment:

        deactivate


Running
-----------------

    python main.py


Testing
-----------------

    python -m unittest discover pdapt_lib 'test_*.py'

To run doc tests, pick a module:

    python -m doctest -v pdapt_lib/machine_learning/maths.py

    python -m doctest -v pdapt_lib/machine_learning/stats.py

    python -m doctest -v pdapt_lib/machine_learning/probs.py

    python -m doctest -v pdapt_lib/machine_learning/cross_validation.py

    python -m doctest -v pdapt_lib/machine_learning/optimize.py

    python -m doctest -v pdapt_lib/machine_learning/regression.py

Documentation
---------------

      cd doc

      make html

      firefox _build/html/index.html


License
---------------

see license.txt file for this



