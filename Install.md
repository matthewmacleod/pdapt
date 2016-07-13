Installation
--------------------

pdapt uses python3 by default


Virtualenv setup
-------------------------------------
      cd

      cd develop

      mkdir pdapt

      cd pdapt

linux:

      virtualenv -p /usr/bin/python3 venv

on mac:

      virtualenv -p /usr/local/bin/python3 venv



Lastly, activate your environment:

      source venv/bin/activate

      pip install --upgrade pip

      pip install numpy

      pip install -r requirements.txt


Documentation setup
-------------------------------------

      easy_install Sphinx


document requirements:

     pip freeze > requirements.txt

      mkdir doc

      ../bin/sphinx-quickstart

  *do not forget to add autodocumentation*

 Edit conf.py to specify sys.path: sys.path.insert(0, os.path.abspath('..'))

      ../bin/sphinx-apidoc -o . ..

      make html

      firefox _build/html/index.html


Additional packages
-------------------------------------

      easy_install ipython

      easy_install "ipython[notebook]"

      easy_install numpy

      easy_install scipy

      easy_install matplotlib

      easy_install nose

      easy_install coverage

      easy_install pyflakes

      easy_install pep8

      pip install seaborn




Addition packages for cross-checking code
-------------------------------------

      easy_install pandas

      easy_install scikit-learn

      pip install Statsmodels

      pip install patsy

      pip install git+https://github.com/pymc-devs/pymc3

      pip install rpy2

      pip install h2o
ann

      pip install theano pyyaml h5py cuDNN

      pip install keras

      pip install lasagne

      pip uninstall nolearn

      pip install  https://github.com/dnouri/nolearn/archive/master.zip#egg=nolearn

      pip install py4j

      pip install six

      TensorFlow:

linux:

      export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp35-cp35m-linux_x86_64.whl
      pip install --upgrade $TF_BINARY_URL

mac:

       export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py3-none-any.whl

       pip install --upgrade $TF_BINARY_URL


webscraping, nlp

      pip install beautifulsoup4 requests python-dateutil twython scrapy

      pip install gensim

      pip install nltk

      pip install crab

      pip install pyldavis

financial

      pip install zipline

      pip install Quandl


