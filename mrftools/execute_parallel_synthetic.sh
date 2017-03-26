#!/bin/bash
for i in 10 20 30 40 50 60 70 80 90 100 150 200 250 300 350 400 450 500
do
  nohup stdbuf -oL python generate_nll_plots.py --nodes_num $i > logs/nll_$i &
done
