#!/bin/bash

if [[ $(poetry config virtualenvs.create) = true ]]
then
    poetry run flake8 src/
else
    flake8 src/
fi
