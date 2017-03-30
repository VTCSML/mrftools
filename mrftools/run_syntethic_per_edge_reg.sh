#!/bin/bash
rm -r ../../../results_compare_nll_edge_reg_new
mkdir ../../../results_compare_nll_edge_reg_new
for nodes_num in 25 50 100 
do
  mkdir ../../../results_compare_nll_edge_reg_new/res_$nodes_num
  for edge_reg in 0.000001 0.0000025 0.000005 0.0000075 0.00001 0.000025 0.00005 0.000075 0.0001 0.00025 0.0005 0.00075 0.001 0.0025 0.005 0.0075 0.01 0.025 0.05 0.075 0.1 0.25 0.5 0.75 1
  do
  	mkdir ../../../results_compare_nll_edge_reg_new/res_$nodes_num/$edge_reg
  	c="$nodes_num _"
  	c=$c$edge_reg
    nohup stdbuf -oL python generate_nll_plots_per_edge_reg.py --nodes_num $nodes_num --edge_reg $edge_reg > logs/nll_"$c" &
  done
done
