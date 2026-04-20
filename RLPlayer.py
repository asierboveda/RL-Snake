import random
import os
import pickle

class RLPlayer:

	def __init__(self, playerID, color, game, epsilon=0.1, alpha=0.1, gamma=0.9):
		self.playerID = playerID
		self.color = color
		self.game = game
		
		# Hiperparámetros de Q-Learning (Markov Decision Process)
		self.epsilon = epsilon # Tasa de exploración
		self.alpha = alpha     # Tasa de aprendizaje (Learning Rate)
		self.gamma = gamma     # Factor de descuento para recompensas futuras
		self.actions = ['N', 'S', 'E', 'W']
		
		self.q_file = f'./q_table_p{playerID}.pkl'
		self.q_table = self.load_q_table()
		
		# Memoria temporal para la actualización de la Q-Table
		self.last_state = None
		self.last_action = None
		self.last_score = 0
		
	def load_q_table(self):
		if os.path.exists(self.q_file):
			with open(self.q_file, 'rb') as f:
				return pickle.load(f)
		return {}
		
	def save_q_table(self):
		with open(self.q_file, 'wb') as f:
			pickle.dump(self.q_table, f)
			
	def get_state(self, my_snake):
		head = my_snake.body[0]
		
		# 1. Detección de Peligros inmediatos (Muros o serpientes)
		danger_N = self.is_dangerous([head[0]-1, head[1]])
		danger_S = self.is_dangerous([head[0]+1, head[1]])
		danger_E = self.is_dangerous([head[0], head[1]+1])
		danger_W = self.is_dangerous([head[0], head[1]-1])
		
		# 2. Dirección cardinal hacia el objetivo (Fruta o Rival)
		goal = self.find_goal(my_snake)
		dir_N = goal[0] < head[0]
		dir_S = goal[0] > head[0]
		dir_E = goal[1] > head[1]
		dir_W = goal[1] < head[1]
		
		# 3. Estado de cazador (Recompensa Battle Royale)
		is_hunter = my_snake.getFruitScore() >= 120
		
		return (danger_N, danger_S, danger_E, danger_W, dir_N, dir_S, dir_E, dir_W, is_hunter)

	def is_dangerous(self, pos):
		# Muros
		if pos[0] < 0 or pos[0] >= self.game.rSize or pos[1] < 0 or pos[1] >= self.game.cSize:
			return True
		# Colisión con el cuerpo de cualquier serpiente (incluida sí misma)
		for s in self.game.snakes:
			if s.isAlive and s.occupies(pos):
				return True
		return False

	def find_goal(self, my_snake):
		ownFruitScore = my_snake.getFruitScore()
		weaker_rivals = [s for i, s in enumerate(self.game.snakes) if i != self.playerID and s.isAlive and s.getFruitScore() < ownFruitScore]
		headPos = my_snake.body[0]
		
		if ownFruitScore >= 120 and weaker_rivals:
			rivalPoss = [p[:2] for s in weaker_rivals for p in s.body]
			if not rivalPoss: return [self.game.rSize//2, self.game.cSize//2]
			return min(rivalPoss, key=lambda p: (headPos[0]-p[0])**2 + (headPos[1]-p[1])**2)
		else:
			fruitPoss = [f.pos for f in self.game.fruits]
			if not fruitPoss: return [self.game.rSize//2, self.game.cSize//2]
			return min(fruitPoss, key=lambda f: (headPos[0]-f[0])**2 + (headPos[1]-f[1])**2)

	def get_q_values(self, state):
		if state not in self.q_table:
			self.q_table[state] = {a: 0.0 for a in self.actions}
		return self.q_table[state]

	def update_q_table(self, state, action, reward, next_state):
		current_q = self.get_q_values(state)[action]
		max_future_q = 0 if next_state is None else max(self.get_q_values(next_state).values())
		new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
		self.q_table[state][action] = new_q

	def get_safe_actions(self, head):
		safe = []
		if not self.is_dangerous([head[0]-1, head[1]]): safe.append('N')
		if not self.is_dangerous([head[0]+1, head[1]]): safe.append('S')
		if not self.is_dangerous([head[0], head[1]+1]): safe.append('E')
		if not self.is_dangerous([head[0], head[1]-1]): safe.append('W')
		return safe

	def play(self, im):
		my_snake = self.game.snakes[self.playerID]
		
		# Si la serpiente murió, penalizamos la acción que la mató (reward negativo gigante)
		if not my_snake.isAlive:
			if self.last_state is not None:
				self.update_q_table(self.last_state, self.last_action, -100, None)
				self.last_state = None
				self.save_q_table()
			return 'N'
			
		current_state = self.get_state(my_snake)
		current_score = my_snake.getScore()
		
		# Calcular recompensa del paso anterior (Aprendizaje Online)
		if self.last_state is not None:
			reward = -0.1 # Penalización leve base para motivar la rapidez y encontrar rutas cortas
			if current_score > self.last_score:
				reward = 20 # Gran recompensa si consiguió fruta o mató rival
			self.update_q_table(self.last_state, self.last_action, reward, current_state)
		
		# Selección de Acción: Política Epsilon-Greedy combinada con Action Masking
		safe_actions = self.get_safe_actions(my_snake.body[0])
		
		if random.random() < self.epsilon:
			action = random.choice(safe_actions if safe_actions else self.actions) # Exploración controlada
		else:
			# Explotación del conocimiento de la Q-Table
			q_values = self.get_q_values(current_state)
			if safe_actions:
				safe_q = {a: q_values[a] for a in safe_actions}
				action = max(safe_q, key=safe_q.get)
			else:
				action = max(q_values, key=q_values.get) # Acción suicida inevitable si no hay safe_actions
				
		self.last_state = current_state
		self.last_action = action
		self.last_score = current_score
		
		# Persistir la memoria frecuentemente
		if self.game.turn % 50 == 0:
			self.save_q_table()
			
		return action