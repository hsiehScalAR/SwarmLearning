import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math
import agent as at
import time
import csv


class Swarm(object):
	def __init__(self, agent_ls):
		self.agent_ls = agent_ls
		self.pos_ls = np.zeros((len(agent_ls),2))
		self.n = len(agent_ls) #total number of agents
		self.it = 0
		
	def move_swarm(self, swarm_file, save=0):
			pos_str = (2*len(agent_ls))*[0]
			for i in range(len(self.agent_ls)):
				agent_ls[i].move_agent(self.pos_ls)
				self.pos_ls[i] = agent_ls[i].pos
				pos_str[i*2] = self.pos_ls[i]
				pos_str[i*2+1] = self.agent_ls[i].vel
			if save==1:
				swarm_writer = csv.writer(swarm_file,delimiter = ',')
				swarm_writer.writerow(pos_str)
			self.it += 1
			print "Iteration:", self.it

if __name__ == '__main__':
	# square initialization
	init_pos = np.array([(np.random.random()*1500, np.random.random()*1500) for i in range(100)])
	init_vel = np.array([10*(0.5-np.array([np.random.random(),np.random.random()])) for i in range(100)])

	# circular initialization
	'''
	def circle_points(r, n):
	    circles = []
	    for r, n in zip(r, n):
	        t = np.linspace(0, 2*np.pi, n)
	        x = r * np.cos(t)
	        y = r * np.sin(t)
	        circles.append(np.c_[x, y])
	    return circles
	init_pos = circle_points([500],[500])[0]
	print len(init_pos)
	for i in range(len(init_pos)):
		init_vel[i][0] = -init_pos[i][1]*0.07
		init_vel[i][1] = init_pos[i][0]*0.07
	'''
	agent_ls = []
	for i in range(len(init_pos)):
		agent_ls.append(at.Agent(init_pos[i],init_vel[i],len(init_pos)))

	swarm = Swarm(agent_ls)

	# initializing animation parameters
	fig = plt.figure()
	ax = plt.axes(xlim=(-3000,3000),ylim=(-3000,3000))
	dots = []
	for i in range(len(swarm.agent_ls)):
		dots.append(swarm.agent_ls[i])
	d,= ax.plot([dot.pos[0] for dot in dots],[dot.pos[1] for dot in dots], c='0.5',marker='.',linestyle='None')

	def animate(i):
		with open('swarm_square_agent100_a5.csv',mode='a') as swarm_file:
			swarm.move_swarm(swarm_file, 0)
	    	d.set_data([dot.pos[0] for dot in dots],[dot.pos[1] for dot in dots])
	    	return d,

	# call the animator.  blit=True means only re-draw the parts that have changed.
	anim = animation.FuncAnimation(fig, animate, frames=200, interval=20)

	plt.show()