Installation
--------------------

pdapt uses python3 by default


Virtualenv setup
-------------------------------------
      cd

      cd develop

      virtualenv -p /usr/bin/python3 pdapt

      cd pdapt

Lastly, activate your environment:

      source bin/activate

Documentation setup
-------------------------------------

      ./bin/easy_install Sphinx


document requirements:

     ./bin/pip3 freeze > requirements.txt

      mkdir doc

      ../bin/sphinx-quickstart

  *do not forget to add autodocumentation*

 Edit conf.py to specify sys.path: sys.path.insert(0, os.path.abspath('..'))

      ../bin/sphinx-apidoc -o . ..

      make html

      firefox _build/html/index.html


Additional packages
-------------------------------------

      ./bin/easy_install ipython

      ./bin/easy_install "ipython[notebook]"

      ./bin/easy_install numpy

      ./bin/easy_install scipy

      ./bin/easy_install matplotlib

      ./bin/easy_install nose

      ./bin/easy_install coverage

      ./bin/easy_install pyflakes

      ./bin/easy_install pep8

      ./bin/pip install seaborn




Addition packages for cross-checking code
-------------------------------------

      ./bin/easy_install pandas

      ./bin/easy_install scikit-learn

      ./bin/pip install Statsmodels

      ./bin/pip install theano patsy

      ./bin/pip install git+https://github.com/pymc-devs/pymc3

