#!/bin/bash
rm -r ../../../results_monitor_ss
mkdir ../../../results_monitor_ss
a="_"
for nodes_num in 50
do
  mkdir ../../../results_monitor_ss/$nodes_num
  for group_l1 in 0.01 0.001
  do
  	mkdir ../../../results_monitor_ss/$nodes_num/$group_l1
  	for l2 in 1.0
  	do
  	  c=$nodes_num
      c=$c$a
  	  c=$c$group_l1
  	  c=$c$a
  	  c=$c$l2
	  mkdir ../../../results_monitor_ss/$nodes_num/$group_l1/$l2
	  nohup stdbuf -oL python monitor_ss.py --nodes_num $nodes_num --group_l1 $group_l1 --l2 $l2 > logs/monitor_ss_"$c" &
	done
  done
done