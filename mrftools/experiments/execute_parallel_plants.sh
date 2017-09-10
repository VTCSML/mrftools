#!/bin/bash
results_folder="results_plants"
rm -r $results_folder
mkdir $results_folder
for edge_num in 200
do
  mkdir $results_folder/$edge_num
  for group_l1 in 0.01 0.001
  do
  	mkdir $results_folder/$edge_num/$group_l1
  	for l2 in 1.0
    do
  	  mkdir $results_folder/$edge_num/$group_l1/$l2
  	  a="_"
  	  c=$edge_num
  	  c=$c$a
  	  c=$c$group_l1
  	  c=$c$a
  	  c=$c$l2
      nohup python plants_experiments.py --edge_num $edge_num --group_l1 $group_l1 --l2 $l2 --results_dir $results_folder  > logs/plants_nll_"$c" &
    done
  done
done