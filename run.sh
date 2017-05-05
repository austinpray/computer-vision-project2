#!/usr/bin/env bash

> out.txt
echo "mnist1.py" | tee -a out.txt
python mnist1.py | tee -a out.txt
echo "mnist2.py" | tee -a out.txt
python mnist2.py | tee -a out.txt
echo "mnist3.py" | tee -a out.txt
python mnist3.py | tee -a out.txt
echo "mnist4.py" | tee -a out.txt
python mnist4.py | tee -a out.txt
echo "mnist5.py" | tee -a out.txt
python mnist5.py | tee -a out.txt
