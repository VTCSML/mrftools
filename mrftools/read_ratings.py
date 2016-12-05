import pandas as pd
import numpy as np
import bisect

RANGE = [-5, 0, 5, 10, 100]

FILES_LIST = ['../../../ratings/jester-data-1.xls', '../../../ratings/jester-data-2.xls', '../../../ratings/jester-data-3.xls'] 

def read_ratings_from_file(file_name, rating_density):
	data = list()
	df = pd.read_excel('../../../ratings/jester-data-1.xls')
	d = df.as_matrix()
	for row_inx in range(d.shape[0]):
		data_instance = dict()
		if d[row_inx][0] > rating_density:
			for col_ind in range(1, d.shape[1]):
				rating = d[row_inx][col_ind]
				mapped_rating = bisect.bisect_left(RANGE, rating)
				data_instance[col_ind] = mapped_rating
			data.append(data_instance)
	return data

def read_ratings_from_batch_files(file_names_list, rating_density):
	data = list()
	for file_name in file_names_list:
		print('>parsing ' + file_name )
		data.extend(read_ratings_from_file(file_name, rating_density))
	print('Done!')
	return data


def main():
	ratings_data = read_ratings_from_batch_files(FILES_LIST, 50)
	print('NUMBER OF DATA INSTANCES:')
	print(len(ratings_data))

if __name__ == '__main__':
	main()


