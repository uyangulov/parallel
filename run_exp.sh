#!/bin/bash

# Запуск main.py с разными значениями n: 50, 100, ..., 300
for n in $(seq 50 50 300); do
    echo "Running for n = $n"
    python3 main.py "$n" 8 5
done
