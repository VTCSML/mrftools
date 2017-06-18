#!/bin/bash
rm -r ../../../results_feature
mkdir ../../../results_feature
a="_"
for nodes_num in 20
do
  mkdir ../../../results_feature/$nodes_num
  for l1 in 0.1 0.01 0.001
  do
  	mkdir ../../../results_feature/$nodes_num/$l1
  	c=$nodes_num
    c=$c$a
  	c=$c$l1
	  nohup stdbuf -oL python generate_nll_feature_graft.py --nodes_num $nodes_num --l1 $l1 > logs/feature_nll_"$c" &
	done
done