#!/bin/bash

watch -n 2 'squeue && echo '' && ls ../data/gnn_records/raw/ | wc -l'

