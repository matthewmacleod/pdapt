#!/usr/bin/env python

# system stuff
import sys
# local stuff
from pdapt_lib.machine_learning import maths

print('using python version = ')
print(sys.version)


print(maths.vector_add([1,1,1],[1,2,3]))
print("testing factorial: ", maths.factorial(1000))
print("testing factorial: ", maths.factorial(100))

