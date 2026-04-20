import random


class RandomPlayer:


	def __init__(self,playerID,color):
		self.playerID=playerID
		self.color=color

	def play(self,im):
		return random.choice(list('NSWE'))