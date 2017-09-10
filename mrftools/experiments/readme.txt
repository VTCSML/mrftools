############################################# Instructions #############################################
+ Running synthetic experiments:

	- Setup the paramter in the bash file execute_parallel_syntehtic.sh:
		- results_folder <the output folder for plots>
		- nodes_num in <list of different numbers of the nodes of each experiment to be ran in parallel>

	- run  "bash execute_parallel_synthetic.sh" command

	- FYI: Results will be shelved under experiments/shelves. If you'd like to change that refer to the script synthetic_experiments.py to redefine the variable "shelve_dir". Make sure the desired directory is already existaing before running the script.
