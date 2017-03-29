#!/bin/bash
for i in 1e-3 2.5e-3 5e-3 7.5e-3 1e-2 2.5e-2 5e-2 7.5e-2 1e-1 2.5e-1 5e-1
do
  nohup stdbuf -oL python generate_nll_ratings.py --edge_reg $i > logs/ratings_nll_$i &
done
