#!/bin/bash
rm -r ../../../results_ratings
mkdir ../../../results_ratings
for edge_num in 100 150 200 300 400 500
do
  mkdir ../../../results_ratings/$edge_num
  for edge_reg in 0.0001 0.001 0.01
  do
  	mkdir ../../../results_ratings/$edge_num/$edge_reg
  	c="$edge_num _"
  	c=$c$edge_reg
    nohup stdbuf -oL python generate_nll_ratings.py --edge_num $edge_num --edge_reg $edge_reg > logs/ratings_nll_"$c" &
  done
done