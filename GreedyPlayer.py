import random


class GreedyPlayer:


	def __init__(self,playerID,color,game):
		self.playerID=playerID
		self.color=color
		self.game=game

	def play(self,im):
		my_snake = self.game.snakes[self.playerID]
		if not my_snake.isAlive:
			return 'N'
			
		ownFruitScore = my_snake.getFruitScore()
		
		# Find weaker rivals
		weaker_rivals = []
		for i, s in enumerate(self.game.snakes):
			if i != self.playerID and s.isAlive:
				if s.getFruitScore() < ownFruitScore:
					weaker_rivals.append(s)

		# Decision
		if ownFruitScore >= 120 and len(weaker_rivals) > 0:
			goal = self.findCloserRival(my_snake, weaker_rivals)
		else:
			goal = self.findCloserFruit(my_snake)
			
		headPos = my_snake.body[0]
		
		direction = self.setDirection(headPos, goal)
		nextPos = headPos.copy()
		if direction=='N':
			nextPos[0] -= 1
		elif direction=='E':
			nextPos[1] += 1
		elif direction=='W':
			nextPos[1] -= 1
		elif direction=='S':
			nextPos[0] += 1
		
		# Simple crash avoidance (only care about living if >= 120)
		if ownFruitScore >= 120:
			crash = False
			for currentPos in my_snake.body:
				if nextPos[:2] == currentPos[:2]:
					crash = True

			if crash:
				direction = random.choice(list(set('NSWE') - set(direction)))
		
		return direction
	
	def findCloserFruit(self, my_snake):
		headPos = my_snake.body[0]
			
		fruitPoss = []
		for f in self.game.fruits:
			fruitPoss.append(f.pos.copy())
			
		if len(fruitPoss) == 0:
			return [self.game.rSize//2, self.game.cSize//2]
			
		minD = 1000000
		goal = fruitPoss[0]
		for f in fruitPoss:
			d = ((headPos[0]-f[0])**2 + (headPos[1]-f[1])**2)**0.5
			if d < minD:
				minD = d
				goal = f
		
		return goal

	def findCloserRival(self, my_snake, weaker_rivals):
		headPos = my_snake.body[0]
			
		rivalPoss = []
		for s in weaker_rivals:
			for p in s.body:
				rivalPoss.append(p[:2])
			
		if len(rivalPoss) == 0:
			return [self.game.rSize//2, self.game.cSize//2]

		minD = 1000000
		goal = rivalPoss[0]
		for p in rivalPoss:
			d = ((headPos[0]-p[0])**2 + (headPos[1]-p[1])**2)**0.5
			if d < minD:
				minD = d
				goal = p
				
		return goal

	def setDirection(self, headPos, goal):
		if goal[0] > headPos[0]:
			return 'S'
		elif goal[0] < headPos[0]:
			return 'N'
		elif goal[1] > headPos[1]:
			return 'E'
		else:
			return 'W'