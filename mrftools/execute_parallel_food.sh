#!/bin/bash
for edge_reg in 1e-2 2.5e-2 5e-2 7.5e-2 1e-1 2.5e-1 5e-1
do
    nohup stdbuf -oL python generate_nll_plots_food.py --edge_reg $edge_reg > logs/nll_$edge_reg &
done
