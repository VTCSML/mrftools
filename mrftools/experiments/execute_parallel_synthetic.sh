#!/bin/bash
results_folder="results_synthetic"
#rm -r $results_folder
mkdir -p $results_folder
a="_"
for nodes_num in 600
do
  mkdir $results_folder/$nodes_num
  for group_l1 in 0.1
  do
  	mkdir $results_folder/$nodes_num/$group_l1
  	for l2 in 1.0
  	do
  	  c=$nodes_num
      c=$c$a
  	  c=$c$group_l1
  	  c=$c$a
  	  c=$c$l2
	  mkdir $results_folder/$nodes_num/$group_l1/$l2
    nohup stdbuf -oL python synthetic_experiments.py --nodes_num $nodes_num --group_l1 $group_l1 --l2 $l2 --results_dir $results_folder > logs/nll_"$c" &
	done
  done
done
