Installation
--------------------

pdapt uses python3 by default


Virtualenv setup
-------------------------------------
      cd

      cd develop

      mkdir pdapt

      cd pdapt

      virtualenv -p /usr/bin/python3 venv
    
on mac:

      virtualenv -p /usr/local/bin/python3 venv
      


Lastly, activate your environment:

      source venv/bin/activate

      pip install --upgrade pip

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

      ./bin/pip install patsy

      ./bin/pip install git+https://github.com/pymc-devs/pymc3

      ./bin/pip install rpy2

      ./bin/pip install h2o
ann

      ./bin/pip install theano pyyaml h5py cuDNN

      ./bin/pip install keras

      ./bin/pip install lasagne

      ./bin/pip uninstall nolearn

      ./bin/pip install  https://github.com/dnouri/nolearn/archive/master.zip#egg=nolearn

      ./bin/pip install py4j

      ./bin/pip install six

      TensorFlow:

      linux:

      export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp35-cp35m-linux_x86_64.whl
      pip install --upgrade $TF_BINARY_URL

      mac
       export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py3-none-any.whl

       pip install --upgrade $TF_BINARY_URL


webscraping, nlp

      ./bin/pip install beautifulsoup4 requests python-dateutil twython scrapy

      ./bin/pip install gensim

      ./bin/pip install nltk

      ./bin/pip install crab

      ./bin/pip install pyldavis

financial

      ./bin/pip install zipline

      ./bin/pip install Quandl


