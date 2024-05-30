#!/bin/bash

# Define variables
T=20
S=10
m1=2
m2=1

# Activate your virtual environment if necessary
# source /path/to/your/venv/bin/activate

# Run your Python script and redirect output to a file
python -u main.py  & $T $S $m1 $m2 > output.log 2>&1 & 
