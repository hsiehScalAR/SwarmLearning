import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib import animation
import sys


my_data = []
with open(sys.argv[1]) as csvDataFile:
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

print "Number of time steps: ", len(pos_from_file)
# Animating a single agent, assuming agents live on a 2D plane
# initializing animation parameters
fig = plt.figure()
global time_step
time_step = 0
ax = plt.axes(xlim=(-3000,3000),ylim=(-3000,3000))
def set_time_step():
	global time_step
	dots = pos_from_file[time_step]
	time_step+=1
	#print time_step
	return dots

dots = set_time_step()
d,= ax.plot([dot[0] for dot in dots],[dot[1] for dot in dots], c='0.5',marker='.',linestyle='None')

def animate(i):
	try:
		dots = set_time_step()
		d.set_data([dot[0] for dot in dots],[dot[1] for dot in dots])
	except:
		print "animation completed"
	return d,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=200, interval=20)

plt.show()
