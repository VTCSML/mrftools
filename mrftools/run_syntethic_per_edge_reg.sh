#!/bin/bash
rm -r ../../../results_compare_nll_edge_reg_new
mkdir ../../../results_compare_nll_edge_reg_new
for nodes_num in 25 50
do
  mkdir ../../../results_compare_nll_edge_reg_new/res_$nodes_num
  for edge_reg in 1e-6 2.5e-6 5e-6 7.5e-6 1e-5 2.5e-5 5e-5 7.5e-5 1e-4 2.5e-4 5e-4 7.5e-4 1e-3 2.5e-3 5e-3 7.5e-3 1e-2 2.5e-2 5e-2 7.5e-2 1e-1 2.5e-1 5e-1 7.5e-1 1
  do
  	mkdir ../../../results_compare_nll_edge_reg_new/res_$nodes_num/$edge_reg
  	c="$nodes_num _"
  	c=$c$edge_reg
    nohup stdbuf -oL python generate_nll_plots_per_edge_reg.py --nodes_num $nodes_num --edge_reg $edge_reg > logs/nll_"$c" &
  done
done
