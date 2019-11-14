import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib import animation

def parse_csv(file_name):
	my_data = []
	with open(file_name) as csvDataFile:
		csvReader = csv.reader(csvDataFile)
		for row in csvReader:
			my_data.append(row)

	#my_data = np.genfromtxt('swarm_ring_500_a1.csv', delimiter=',')
	pos_from_file = []
	line = []
	for i in range(len(my_data)):
		for j in range(len(my_data[0])/2):
			line_str = my_data[i][j*2].lstrip('[').rstrip(']').split()
			line.append([float(line_str[0]), float(line_str[1])])
		pos_from_file.append(line)
		line = []

	return pos_from_file