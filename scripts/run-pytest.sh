#!/bin/bash

if [[ $(poetry config virtualenvs.create) = true ]]
then
    poetry run pytest
else
    pytest
fi
