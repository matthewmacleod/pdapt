#!/bin/bash

# simple script to run pdapt tests

if [ $# -eq 0 ] ; then
  echo "No arguments supplied; give args: unit, doctest, doctestf"
  exit
fi

test=$1

modules="maths stats probs cross_validation optimize regression classification nlp"

. ./venv/bin/activate

if [ $test == 'unit' ]; then
    python -m unittest discover pdapt_lib 'test_*.py'
elif [ $test == 'doctest' ]; then
    for i in `echo $modules`; do
      python -m doctest -v pdapt_lib/machine_learning/"$i".py
    done
elif [ $test == 'doctestf' ]; then
    for i in `echo $modules`; do
      python -m doctest -v pdapt_lib/machine_learning/"$i".py | grep -A 5 Failed
    done
else
    echo "arguement not recognized"
fi

