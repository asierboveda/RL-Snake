import numpy as np

class Fruit:

	def __init__(self,posR,posC,size,value):
		self.pos=[posR,posC]
		self.value=value
		self.timeLeft=25
		self.size=size
		self.visual=self._load_visual()

	def _load_visual(self):
		try:
			from matplotlib import image
			visual=image.imread("./input/"+str(self.size)+"/fruit"+str(self.value)+".png")
			return visual[:,:,:3]
		except Exception:
			return np.zeros((self.size, self.size, 3), dtype=float)

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
	
