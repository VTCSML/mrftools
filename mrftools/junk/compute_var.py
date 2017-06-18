def compute_var(n, s):
	edge_feature = (float((n ** 2 - n)) / 2) * s ** 2 
	node_feature = (n * s)
	tot_feature = edge_feature + node_feature
	print 'edge_feature:', edge_feature
	print 'node_feature:', node_feature
	print 'tot_feature:', tot_feature 

def main():
	var = compute_var(50, 5)

if __name__ == '__main__':
	main()