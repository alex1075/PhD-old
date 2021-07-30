#!/bin/bash
echo 'Make sure you have conda installed'

conda creaate -n ML_workspace tensorflow-gpu==2.4.1

pause

conda activate ML_workspace
pip install -r requirements.txt
