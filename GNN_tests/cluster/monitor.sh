#!/bin/bash

watch -n 2 'squeue && echo '/raw files:' && ls ../data/gnn_records/raw/ | wc -l && echo '/processed files:' && ls ../data/gnn_records/processed/ | wc -l'
