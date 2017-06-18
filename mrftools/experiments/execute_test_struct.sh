#!/bin/bash
rm -r ../../../results_test_struct
mkdir ../../../results_test_struct
a="_"
for nodes_num in 50
do
  mkdir ../../../results_test_struct/$nodes_num
  for group_l1 in 0.0001 0.001 0.01
  do
    mkdir ../../../results_test_struct/$nodes_num/$group_l1
    for l2 in 1.0 2.0
    do
      c=$nodes_num
      c=$c$a
      c=$c$group_l1
      c=$c$a
      c=$c$l2
    mkdir ../../../results_test_struct/$nodes_num/$group_l1/$l2
    nohup stdbuf -oL python test_struct.py --nodes_num $nodes_num --group_l1 $group_l1 --l2 $l2 > logs/test_struct_nll_"$c" &
  done
  done
done