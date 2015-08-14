Installation
--------------------

pdapt uses python3 by default


cd
cd develop
virtualenv -p /usr/bin/python3 pdapt

cd pdapt

./bin/easy_install Sphinx
./bin/easy_install ipython

Lastly, activate your environment:

      source bin/activate

document requirements

    ./bin/pip3 freeze > requirements.txt


mkdir doc
../bin/sphinx-quickstart
[add autodocumentation]

Edit conf.py to specify sys.path: sys.path.insert(0, os.path.abspath('..'))

../bin/sphinx-apidoc -o . ..

make html

firefox _build/html/index.html
