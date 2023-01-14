#!/bin/bash

if [[ $(poetry config virtualenvs.create) = true ]]
then
    poetry run black --check .
else
    black --check .
fi
