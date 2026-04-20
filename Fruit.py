
from matplotlib import image

class Fruit:

	def __init__(self,posR,posC,size,value):
		self.pos=[posR,posC]
		self.value=value
		self.timeLeft=25
		self.size=size
		self.visual=image.imread("./input/"+str(size)+"/fruit"+str(self.value)+".png")
		self.visual=self.visual[:,:,:3]

	def overlaps(self,pos):
		if abs(pos[0]-self.pos[0])<=0 and \
			abs(pos[1]-self.pos[1])<=0:
			return True
		else:
			return False

	def __repr__(self):
		return str(self)

	def __str__(self):
		return 'Fruit at pos'+str(self.pos)
	