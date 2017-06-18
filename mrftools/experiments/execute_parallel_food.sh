#!/bin/bash
rm -r ../../../results_food
mkdir ../../../results_food
for edge_num in 400 
do
  mkdir ../../../results_food/$edge_num
  for group_l1 in  0.01
  do
  	mkdir ../../../results_food/$edge_num/$group_l1
  	for l2 in 0.5 1.0
    do
  	  mkdir ../../../results_food/$edge_num/$group_l1/$l2
  	  a="_"
  	  c=$edge_num
  	  c=$c$a
  	  c=$c$group_l1
  	  c=$c$a
  	  c=$c$l2
      nohup stdbuf -oL python generate_nll_plots_food.py --edge_num $edge_num --group_l1 $group_l1 --l2 $l2 > logs/food_nll_"$c" &
    done
  done
done