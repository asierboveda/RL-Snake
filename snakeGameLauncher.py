from PIL import Image
import os

from SnakeGame import *

from matplotlib import pyplot as plt

from RandomPlayer import RandomPlayer
from GreedyPlayer import GreedyPlayer
from RLPlayer import RLPlayer

#General config
imageFolder = './output'
noiseLevel=0.01#noise level \in[0,1]
minScoringForWinning=120
turnLimit=900


#Remove old files
dir_name = "./output/"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".png"):
        os.remove(os.path.join(dir_name, item))


seedVal=1
random.seed(seedVal)

#Game initialization 
sg=SnakeGame()
for _ in range(5):
	sg.addRandomFruit()
sg.setNoise(noiseLevel)

playerA=RLPlayer(0,'G',sg, epsilon=0.0)  # Modo Inferencia: epsilon=0 → sin exploración aleatoria
playerB=GreedyPlayer(1,'B',sg)
playerC=GreedyPlayer(2,'R',sg)
playerD=GreedyPlayer(3,'Y',sg)



while sg.turn <turnLimit and sg.gameIsAlive():

	if sg.turn%10==0:
		 print('\nTurno '+str(sg.turn))

	#Board snapshotting
	board=sg.getSnapshot()
	plt.imshow(board)
	plt.draw()
	plt.pause(0.05)
	plt.clf()

	#Image saving
	b = (board*255).astype(np.uint8)
	image = Image.fromarray(b)
	image.save('./'+imageFolder+'/snake'+str(sg.turn).zfill(3)+'.png')

	#Game
	directionA=playerA.play(board)
	directionB=playerB.play(board)
	directionC=playerC.play(board)
	directionD=playerD.play(board)

	#Turn-based Update
	sg.movePlayer(0, directionA)
	sg.movePlayer(1, directionB)
	sg.movePlayer(2, directionC)
	sg.movePlayer(3, directionD)
	sg.checkMovements()
	sg.update()

	sg.turn+=1
	


#Board snapshotting
board=sg.getFinalSnapshot()
plt.imshow(board, interpolation='nearest')
plt.show()

#Image saving
b = (board*255).astype(np.uint8)
image = Image.fromarray(b)
image.save('./'+imageFolder+'/snake'+str(sg.turn).zfill(3)+'.png')


#Game recap and result printing
print('\n----\nGame finished with score '+str(sg.getScores())+' in '+str(sg.turn)+' turns')

snake_labels = ['A','B','C','D']
for i, s in enumerate(sg.snakes):
	status = 'alive' if s.isAlive else 'dead'
	print(f'\t Snake {snake_labels[i]}: {s.getScore()} pts ({status})')

alive = [(i, s) for i, s in enumerate(sg.snakes) if s.isAlive]
if len(alive) == 1:
	print(f'Winner: Snake {snake_labels[alive[0][0]]}')
elif len(alive) == 0:
	print('Winner: No winner (all dead)')
else:
	scores = sg.getScores()
	max_score = max(scores)
	if max_score < minScoringForWinning:
		print('Winner: No winner (too few points)')
	else:
		winners = [i for i, sc in enumerate(scores) if sc == max_score]
		if len(winners) == 1:
			print(f'Winner: Snake {snake_labels[winners[0]]}')
		else:
			print('Winner: No winner (draw)')
