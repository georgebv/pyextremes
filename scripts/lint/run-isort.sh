#!/bin/bash

if [[ $(poetry config virtualenvs.create) = true ]]
then
    poetry run isort --check .
else
    isort --check .
fi
