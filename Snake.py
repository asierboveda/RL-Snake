import numpy as np


def _load_image(path, fallback_shape=(17, 17, 3)):
	try:
		from matplotlib import image
		pic = image.imread(path)
		return pic[:,:,:3]
	except Exception:
		return np.zeros(fallback_shape, dtype=float)

class Snake:
	
	def __init__(self,number,color,tile,game):
		self.playerNumber=number
		self.color=color
		self.body=[tile]
		self.game=game
		self.isAlive=True
		self.fruitEaten=False
		self.points=0
		self.fruit_points=0


	def move(self,direction):

		newPosition=self.body[0].copy()
		
		if direction=='N':
			newPosition[0]=newPosition[0]-1
		elif direction=='E':
			newPosition[1]=newPosition[1]+1
		elif direction=='W':
			newPosition[1]=newPosition[1]-1
		elif direction=='S':
			newPosition[0]=newPosition[0]+1
		newPosition[2]=direction

		fruitEaten=False
		for f in list(self.game.fruits):
			if f.overlaps(newPosition):
				fruitEaten=True
				
		self.fruitEaten=fruitEaten
				
		if fruitEaten:
			self.body.insert(0,newPosition)
		else:
			self.body.insert(0,newPosition)
			self.body=self.body[:-1]

	def getBody(self):
		return self.body

	def headPos(self):
		return self.body[0].copy()

	def addPoints(self,points,from_fruit=False):
		self.points=self.points+points
		if from_fruit:
			self.fruit_points=self.fruit_points+points

	def getFruitScore(self):
		return self.fruit_points

	def occupies(self,pos):
		for piece in self.body:
			if piece[:2] == pos[:2]:
				return True
		return False

	def getPositions(self):
		return self.body
	
	def isOutOfBounds(self):
		
		#Should only happen with the head, but we check it all for future robustness
		for piece in self.body:
			if piece[0]<0 or piece[0]>=self.game.rSize or \
				piece[1]<0 or piece[1]>=self.game.cSize:
				return True
		return False
	
	def eatItself(self):
		#Should only happen with the head, as game should stop right away
		if len(self.body)>1:
			for p in self.body[1:]:
				if self.body[0][:2] == p[:2]:
					 return True
		return False
	 
	def __repr__(self):
		return 'Snake at'+str(self.body)
	 
	def snakeAlive(self):
		return self.isAlive
	
	def drawYourself(self,im):
		
		if not self.isAlive:
			return # No dibujarse si esta muerta
			
		self.drawHead(im)#head
		self.drawBody(im)
		
		
	def drawHead(self,im):
		
		pos=self.body[0]
		
		fromR=(pos[0]+self.game.upperPanelHeight)*(self.game.SFACTOR+1)+1
		
		toR=fromR+self.game.SFACTOR
		fromC=(pos[1])*(self.game.SFACTOR+1)+1
		toC=fromC+self.game.SFACTOR

		pic=_load_image("./input/"+str(self.game.SFACTOR)+"/snakehead_"+self.color+"_"+pos[2]+".png")
			
		#we need to check in case snake is dead by getting out of limits
		if not self.snakeAlive():
			if fromR>=0 and fromC>=0 and toR<im.shape[0] and toC<im.shape[1]:
				im[fromR:toR,fromC:toC,:]=pic
		else:
			im[fromR:toR,fromC:toC,:]=pic
			
	def drawBody(self,im):
		
		for piecePos in range(1,len(self.body)-1):#body
			piece=self.body[piecePos].copy()
			prevPiece=self.body[piecePos-1].copy()
			if piece[2]==prevPiece[2]:
				self.drawBodyPiece(piece,im)
			else:
				piece[2]=piece[2]+prevPiece[2]
				
				if 'E' in piece[2] and 'W' in piece[2]:
					#Snake went backwards and that's a death. Undrawable
					piece[2]=piece[2][0]
				elif 'N' in piece[2] and 'S' in piece[2]:
					#Same, just vertically
					piece[2]=piece[2][0]
				
				
				self.drawBodyPiece(piece,im)
		
		if len(self.body)>1:
			self.drawTail(im)
			
	def getScore(self):
		return self.points
			
			
	def drawBodyPiece(self,piece,im):
		
		pos=piece
		
		fromR=(pos[0]+self.game.upperPanelHeight)*(self.game.SFACTOR+1)+1
		toR=fromR+self.game.SFACTOR
		fromC=(pos[1])*(self.game.SFACTOR+1)+1
		toC=fromC+self.game.SFACTOR

		pic=_load_image("./input/"+str(self.game.SFACTOR)+"/snake_"+self.color+"_"+pos[2]+".png")
		im[fromR:toR,fromC:toC,:]=pic

	def drawTail(self,im):
		
		pos=self.body[-1].copy()
		prevPos=self.body[-2].copy()
		pos[2]=prevPos[2]
		
		
		fromR=(pos[0]+self.game.upperPanelHeight)*(self.game.SFACTOR+1)+1
		toR=fromR+self.game.SFACTOR
		fromC=(pos[1])*(self.game.SFACTOR+1)+1
		toC=fromC+self.game.SFACTOR

		pic=_load_image("./input/"+str(self.game.SFACTOR)+"/snaketail_"+self.color+"_"+pos[2]+".png")
		im[fromR:toR,fromC:toC,:]=pic	
			
