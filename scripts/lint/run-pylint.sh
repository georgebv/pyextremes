#!/bin/bash

if [[ $(poetry config virtualenvs.create) = true ]]
then
    poetry run pylint src/ --rcfile=pyproject.toml
else
    pylint src/ --rcfile=pyproject.toml
fi
