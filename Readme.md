Pdapt
==============================

python machine learning suite coded from scratch


Using virtualenv

       source/bin/activate

to deactivate environment

        deactivate


Run
-----------------

     python main.py


Testing
-----------------

     python -m unittest discover pdapt_lib 'test_*.py'

to run doc tests:

    python -m doctest -v pdapt_lib/machine_learning/maths.py

Documentation
---------------

      cd doc

      make html

      firefox _build/html/index.html

