import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation
import time
import math
#random distribution scaled by 0.1
#a 0.25

class Agent(object):
	def __init__(self, pos, vel, agent_count, coupling=5, delay=0, conn_num=55, conn_arr=[]):
		self.pos = pos
		self.vel = vel
		self.coupling = coupling
		self.delay = delay
		self.conn_num = conn_num
		self.agent_count = agent_count
		self.conn_arr = np.array(np.random.randint(0,agent_count,conn_num))
		print self.conn_arr
		self.acc = 0.0

	def update_acc(self, vel, pos_ls=[0.0, 1.0]):
		#f = (np.sum(np.tile(self.pos,(self.agent_count,1)),axis=0)-
		#	np.sum(pos_ls,axis=0))*self.coupling/(self.agent_count) #swarm coupling term
		#Limited connectivity
		f = (np.sum((np.tile(self.pos,(self.conn_num,1))),axis=0)-
			np.sum(pos_ls[self.conn_arr],axis=0))*self.coupling/(self.conn_num) #swarm coupling term
		#print "mean field: ", np.sum(pos_ls,axis=0)*a/(agents_count)
		p = (1-np.dot(vel,vel))*vel #self-propulsion term
		n = np.random.normal() #noise term
		acc = p-f+n*1 #acceleration without delay term
		return acc


	def move_agent(self,pos_ls=[2.0,1.0]):
		t_0 = time.time()
		self.acc = self.update_acc(self.vel,pos_ls)
		t_1 = time.time()
		self.vel = self.vel+self.acc*0.00005
		self.pos = self.pos+self.vel
		#print "vel: ", self.vel
		#print "pos: ", self.pos


if __name__ == '__main__':
	# Animating a single agent, assuming agents live on a 2D plane
	fig = plt.figure()
	ax = plt.axes(xlim=(-1000,1000),ylim=(-1000,1000))
	dot = Agent(np.array([2.0,1.0]))
	d,= ax.plot(dot.pos[0],dot.pos[1], 'r')

	def animate(i):
	    dot.move_agent()
	    d.set_data(dot.pos[0],dot.pos[1])
	    return d,

	# call the animator.  blit=True means only re-draw the parts that have changed.
	anim = animation.FuncAnimation(fig, animate, frames=200, interval=20)
	plt.show()
