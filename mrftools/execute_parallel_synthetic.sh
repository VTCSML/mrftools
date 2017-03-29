#!/bin/bash
for i in 25 50 100 200 400
do
  nohup stdbuf -oL python generate_nll_plots.py --nodes_num $i > logs/nll_$i &
done
