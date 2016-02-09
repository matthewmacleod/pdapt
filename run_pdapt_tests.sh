#!/usr/bin/bash

# simple script to run pdapt tests

if [ $# -eq 0 ] ; then
  echo "No arguments supplied; give args: unit, doctest, doctestf"
  exit
fi

test=$1

. ./bin/activate

if [ $test == 'unit' ]; then
    python -m unittest discover pdapt_lib 'test_*.py'
elif [ $test == 'doctest' ]; then
    python -m doctest -v pdapt_lib/machine_learning/maths.py
    python -m doctest -v pdapt_lib/machine_learning/stats.py
    python -m doctest -v pdapt_lib/machine_learning/probs.py
    python -m doctest -v pdapt_lib/machine_learning/cross_validation.py
    python -m doctest -v pdapt_lib/machine_learning/optimize.py
    python -m doctest -v pdapt_lib/machine_learning/regression.py
    python -m doctest -v pdapt_lib/machine_learning/nlp.py
elif [ $test == 'doctestf' ]; then
    python -m doctest -v pdapt_lib/machine_learning/maths.py | grep -A 5 Failed
    python -m doctest -v pdapt_lib/machine_learning/stats.py | grep -A 5 Failed
    python -m doctest -v pdapt_lib/machine_learning/probs.py | grep -A 5 Failed
    python -m doctest -v pdapt_lib/machine_learning/cross_validation.py | grep -A 5 Failed
    python -m doctest -v pdapt_lib/machine_learning/optimize.py | grep -A 5 Failed
    python -m doctest -v pdapt_lib/machine_learning/regression.py | grep -A 5 Failed
    python -m doctest -v pdapt_lib/machine_learning/nlp.py | grep -A 5 Failed
else
    echo "arguement not recognized"
fi

