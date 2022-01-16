#!/bin/bash

rm -rf build dist hyperbox.egg-info
python setup.py sdist bdist_wheel
echo Building...
echo Uploading...
if (($1==1));then
    python -m twine upload dist/*
    echo Uploading to PyPI... 
elif (($1==2));then
    python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* 
    echo Uploading to TestPyPI...
else
    echo "Wrong command, only support 1 or 2"
fi
echo Done.