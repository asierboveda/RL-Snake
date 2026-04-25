import numpy as np
import random


from Snake import *
from Fruit import *



class SnakeGame:

	def __init__(self):
		self.rSize=44
		self.cSize=44
		self.snakes=[
			Snake(0,'G',[10,10,'N'],self),
			Snake(1,'B',[10,34,'N'],self),
			Snake(2,'R',[34,10,'N'],self),
			Snake(3,'Y',[34,34,'N'],self)
		]
		self.fruits=[]

		self.dir='N'
		self.numTurns=0
		self.SFACTOR=17
		self.gameAlive=True
		self.noise=0
		self.upperPanelHeight=6
		self.turn=0

	def setNoise(self,n):
		self.noise=n

	def __repr__(self):
		st='Snake Game\n'
		for i, s in enumerate(self.snakes):
			st+='Snake '+str(i)+' Positions:\n'
			for pos in s.getPositions():
				st+='\t'+str(pos)+'\n'
		st+='Fruits:'
		for f in self.fruits:
			st+='\t'+str(f)+'\n'

		return st

	def addRandomFruit(self):
		if self.turn>30 and len(self.snakes) > 0:
			# Get a random alive snake
			alive_snakes = [s for s in self.snakes if s.isAlive]
			if alive_snakes:
				chosen_s = random.choice(alive_snakes)
				posR=chosen_s.body[0][0]
				posC=chosen_s.body[0][1]
			else:
				posR=random.randint(0,self.rSize-1)
				posC=random.randint(0,self.cSize-1)
		else:
			posR=random.randint(0,self.rSize-1)
			posC=random.randint(0,self.cSize-1)
			
		while any(s.occupies([posR,posC]) for s in self.snakes) or self.thereIsFruitAt([posR,posC]):
			if random.random()<0.2:
				alive_snakes = [s for s in self.snakes if s.isAlive]
				if alive_snakes:
					chosen_s = random.choice(alive_snakes)
					posR=chosen_s.body[0][0]
					posC=chosen_s.body[0][1]
				else:
					posR=random.randint(0,self.rSize-1)
					posC=random.randint(0,self.cSize-1)
			else:
				posR=random.randint(0,self.rSize-1)
				posC=random.randint(0,self.cSize-1)

		self.fruits.append(Fruit(posR,posC,self.SFACTOR,random.choice([10,15,20])))

	def thereIsFruitAt(self,pos):
		for f in self.fruits:
			if f.overlaps(pos):
				return True
		return False

	def getScores(self):
		return [s.getScore() for s in self.snakes]
	
	def update(self):
		#nuevas frutas
		if self.numTurns % 5==0:
			self.addRandomFruit()

		#frutas caducadas
		for f in list(self.fruits):
			f.timeLeft=f.timeLeft-1
			if f.timeLeft==0:
				self.fruits.remove(f)

		self.numTurns=self.numTurns+1

	def movePlayer(self, playerID, direction):
		if self.snakes[playerID].isAlive:
			self.snakes[playerID].move(direction)

	def checkMovements(self):
		
		#Caso 1: Chequear serpientes que se salen
		for s in self.snakes:
			if s.isAlive and s.isOutOfBounds():
				s.isAlive=False

		#Caso 2: Serpiente se choca consigo misma
		for s in self.snakes:
			if s.isAlive and s.eatItself():
				s.isAlive=False

		#Caso 3/4 Battle Royale: Choques entre serpientes
		# Reglas: "Mata siempre la que tiene mas puntos (con la condicion de mas de 120)"
		# "si no tienen esos puntos y se chocan(da igual la parte la cuerpo) se mueren las dos"
		# Kill adds 30 points.
		snakes_to_kill = set()
		killer_awards = []
		
		for i in range(len(self.snakes)):
			for j in range(i+1, len(self.snakes)):
				s1 = self.snakes[i]
				s2 = self.snakes[j]
				if not s1.isAlive and not s2.isAlive:
					continue
				
				# Check collision (any part overlaps)
				crashed = False
				for p1 in s1.body:
					for p2 in s2.body:
						if p1[:2] == p2[:2]:
							crashed = True
							break
					if crashed:
						break
				
				if crashed:
					# Verify rules based on fruit points
					pts_fruit1 = s1.getFruitScore()
					pts_fruit2 = s2.getFruitScore()
					
					if pts_fruit1 < 120 and pts_fruit2 < 120:
						snakes_to_kill.add(i)
						snakes_to_kill.add(j)
					elif pts_fruit1 == pts_fruit2:
						snakes_to_kill.add(i)
						snakes_to_kill.add(j)
					else:
						if pts_fruit1 > pts_fruit2:
							snakes_to_kill.add(j) # s2 dies
							if s1.isAlive and i not in snakes_to_kill:
								killer_awards.append(i) # s1 gets points
						elif pts_fruit2 > pts_fruit1:
							snakes_to_kill.add(i) # s1 dies
							if s2.isAlive and j not in snakes_to_kill:
								killer_awards.append(j) # s2 gets points
						
		# Apply deaths and awards
		for idx in snakes_to_kill:
			self.snakes[idx].isAlive = False
			
		for idx in killer_awards:
			if self.snakes[idx].isAlive:
				self.snakes[idx].addPoints(30, from_fruit=False)


        # Caso frutas comidas
		for snake in self.snakes:
			if snake.snakeAlive() and snake.fruitEaten:
				removeIndex = -1
				for idxF in range(0,len(list(self.fruits))):
					f=self.fruits[idxF]
					if f.overlaps(snake.headPos()):
						fruitPoints=f.value
						removeIndex=idxF
						break
				
				if removeIndex != -1:
					snake.addPoints(fruitPoints, from_fruit=True)
					del self.fruits[removeIndex]
				
		#Recap: Game is alive if > 1 snake is alive
		alive_count = sum([1 for s in self.snakes if s.isAlive])
		self.gameAlive = (alive_count > 1)

	def gameIsAlive(self):
		return self.gameAlive

	def getSnapshot(self,final=False):

		image=0.8*np.ones([((self.rSize+self.upperPanelHeight)*(self.SFACTOR+1))+1,(self.cSize*(self.SFACTOR+1))+1,3]).astype(float)

		for row in range(0,self.rSize):
			for col in range(0,self.cSize):
				self.fillCell(image,[row,col],[0.9,0.9,0.9])

		#draw upper panel
		self.drawUpperPanel(image)
		self.drawTimer(image)

		for s in self.snakes:
			s.drawYourself(image)
				
		#draw fruits
		for f in self.fruits:
			self.drawFruit(image,f)

		#add noise
		gaussianNoise = np.random.normal(0, self.noise, image.shape)
		image=image+gaussianNoise
		image[image<0]=0
		image[image>1]=1

		return image

	def getFinalSnapshot(self):
		return self.getSnapshot(True)

	def drawUpperPanel(self,im):
		colors = ['g','b','r','y']
		# Spread the scores to the sides to leave the center open for the timer
		positionsC = [2, 10, 28, 36]
		for i, s in enumerate(self.snakes):
			posC = positionsC[i]
			self.drawScore(im, s.getScore(), 2, posC, colors[i])
		
	def drawTimer(self,im):
		color='g'
		# Center the timer securely
		self.drawScore(im,self.numTurns,1,self.cSize//2 - 3,color)
		
	def drawScore(self,im,score,posR,posC,color):
		
		score=str(score).zfill(3)
		scoreDigits=[int(score[0]),int(score[1]),int(score[2])]
		
		for numDigit in [0,1,2]:
			for offSetR in [0,1,2]:
				for offSetC in [0,1]:
					fromR=(posR+offSetR)*(self.SFACTOR+1)+1
					toR=fromR+self.SFACTOR
					fromC=(posC+(2*numDigit)+offSetC)*(self.SFACTOR+1)+1
					toC=fromC+self.SFACTOR
					
					try:
						from matplotlib import image
						pic=image.imread("./input/"+str(self.SFACTOR)+"/"+str(scoreDigits[numDigit])+color+"-"+str(offSetR)+"-"+str(offSetC)+".png")
						pic=pic[:,:,:3]
						if fromR>=0 and fromC>=0 and toR<im.shape[0] and toC<im.shape[1]:
							im[fromR:toR,fromC:toC,:]=pic
					except Exception:
						pass


	def drawFruit(self,im,f):
		pos=f.pos
		fromR=(pos[0]+self.upperPanelHeight)*(self.SFACTOR+1)+1
		toR=fromR+self.SFACTOR
		fromC=(pos[1])*(self.SFACTOR+1)+1
		toC=fromC+self.SFACTOR
		im[fromR:toR,fromC:toC,:]=f.visual

	def fillCell(self,im,pos,color):
		fromR=(pos[0]+self.upperPanelHeight)*(self.SFACTOR+1)+1
		toR=fromR+self.SFACTOR
		fromC=(pos[1])*(self.SFACTOR+1)+1
		toC=fromC+self.SFACTOR
		im[fromR:toR,fromC:toC,:]=color

