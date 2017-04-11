#!/bin/bash
rm -r ../../../results_compare_nll_edge_reg_new
mkdir ../../../results_compare_nll_edge_reg_new
a="_"
for nodes_num in 50
do
  mkdir ../../../results_compare_nll_edge_reg_new/$nodes_num
  for group_l1 in 0.01 0.001
  do
  	mkdir ../../../results_compare_nll_edge_reg_new/$nodes_num/$group_l1
  	for l2 in 0.5 0.75
  	do
  	  c=$nodes_num
      c=$c$a
  	  c=$c$group_l1
  	  c=$c$a
  	  c=$c$l2
	  mkdir ../../../results_compare_nll_edge_reg_new/$nodes_num/$group_l1/$l2
	  nohup stdbuf -oL python generate_nll_plots_per_edge_reg.py --nodes_num $nodes_num --group_l1 $group_l1 --l2 $l2 > logs/nll_"$c" &
	done
  done
done