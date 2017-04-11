#!/bin/bash
rm -r ../../../results_plants
mkdir ../../../results_plants
for edge_num in 125 175
do
  mkdir ../../../results_plants/$edge_num
  for group_l1 in 0.01 0.001 0.0001
  do
  	mkdir ../../../results_plants/$edge_num/$group_l1
  	for l2 in 0.5 1.0 1.5
    do
  	  mkdir ../../../results_plants/$edge_num/$group_l1/$l2
  	  a="_"
  	  c=$edge_num
  	  c=$c$a
  	  c=$c$group_l1
  	  c=$c$a
  	  c=$c$l2
      nohup stdbuf -oL python generate_nll_plants.py --edge_num $edge_num --group_l1 $group_l1 --l2 $l2 > logs/plants_nll_"$c" &
    done
  done
done