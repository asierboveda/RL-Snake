import random
import os
import pickle

from board_state import (
	BOARD_COLS,
	BOARD_ROWS,
	BoardState,
	FruitState,
	SnakeState,
	determine_winner,
)

class RLPlayer:

	def __init__(self, playerID, color, game, epsilon=0.1, alpha=0.1, gamma=0.9, training_enabled=True):
		self.playerID = playerID
		self.color = color
		self.game = game
		self.training_enabled = training_enabled  # False = modo evaluacion (sin actualizaciones Q)
		
		# Hiperparámetros de Q-Learning (Markov Decision Process)
		self.epsilon = epsilon # Tasa de exploración
		self.alpha = alpha     # Tasa de aprendizaje (Learning Rate)
		self.gamma = gamma     # Factor de descuento para recompensas futuras
		self.actions = ['N', 'S', 'E', 'W']
		
		import os
		os.makedirs('models', exist_ok=True)
		self.q_file = f'models/q_table_p{playerID}.pkl'
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

	def board_state_from_game(self):
		if self.game is None:
			raise ValueError("RLPlayer has no game adapter; pass a BoardState to play_board_state()")

		labels = ("A", "B", "C", "D")
		snakes = []
		for idx, snake in enumerate(self.game.snakes):
			body = tuple(
				(piece[0], piece[1], piece[2])
				for piece in snake.body
				if 0 <= piece[0] < BOARD_ROWS and 0 <= piece[1] < BOARD_COLS
			)
			if not body:
				body = ((0, 0, "N"),)
			snakes.append(
				SnakeState(
					player_id=idx,
					label=labels[idx],
					color=snake.color,
					alive=snake.isAlive,
					body=body,
					score=snake.getScore(),
					fruit_score=snake.getFruitScore(),
				)
			)

		winner_id, terminal_reason = determine_winner(snakes)
		game_alive = self.game.gameIsAlive()
		return BoardState(
			turn=self.game.turn,
			rows=BOARD_ROWS,
			cols=BOARD_COLS,
			snakes=tuple(snakes),
			fruits=tuple(
				FruitState(row=fruit.pos[0], col=fruit.pos[1], value=fruit.value, time_left=fruit.timeLeft)
				for fruit in self.game.fruits
			),
			game_alive=game_alive,
			winner_id=None if game_alive else winner_id,
			terminal_reason=None if game_alive else terminal_reason,
		)

	def get_my_snake(self, board_state):
		for snake in board_state.snakes:
			if snake.player_id == self.playerID:
				return snake
		raise ValueError(f"player_id {self.playerID} not present in BoardState")

	def get_state_from_board(self, board_state):
		my_snake = self.get_my_snake(board_state)
		head = my_snake.head

		danger_N = self.is_dangerous_on_board((head[0]-1, head[1]), board_state)
		danger_S = self.is_dangerous_on_board((head[0]+1, head[1]), board_state)
		danger_E = self.is_dangerous_on_board((head[0], head[1]+1), board_state)
		danger_W = self.is_dangerous_on_board((head[0], head[1]-1), board_state)

		goal = self.find_goal_on_board(board_state)
		dir_N = goal[0] < head[0]
		dir_S = goal[0] > head[0]
		dir_E = goal[1] > head[1]
		dir_W = goal[1] < head[1]

		return (danger_N, danger_S, danger_E, danger_W, dir_N, dir_S, dir_E, dir_W, my_snake.is_hunter)

	def is_dangerous_on_board(self, pos, board_state):
		row, col = pos
		if row < 0 or row >= board_state.rows or col < 0 or col >= board_state.cols:
			return True
		for snake in board_state.snakes:
			if snake.alive and (row, col) in snake.occupied_cells():
				return True
		return False

	def find_goal_on_board(self, board_state):
		my_snake = self.get_my_snake(board_state)
		head = my_snake.head
		weaker_rivals = [
			snake for snake in board_state.snakes
			if snake.player_id != self.playerID and snake.alive and snake.fruit_score < my_snake.fruit_score
		]

		if my_snake.is_hunter and weaker_rivals:
			rival_positions = [cell for snake in weaker_rivals for cell in snake.occupied_cells()]
			return min(rival_positions, key=lambda pos: (head[0]-pos[0])**2 + (head[1]-pos[1])**2)

		fruit_positions = [(fruit.row, fruit.col) for fruit in board_state.fruits]
		if not fruit_positions:
			return (board_state.rows//2, board_state.cols//2)
		return min(fruit_positions, key=lambda pos: (head[0]-pos[0])**2 + (head[1]-pos[1])**2)

	def get_safe_actions_from_board(self, board_state):
		my_snake = self.get_my_snake(board_state)
		head = my_snake.head
		candidates = {
			'N': (head[0]-1, head[1]),
			'S': (head[0]+1, head[1]),
			'E': (head[0], head[1]+1),
			'W': (head[0], head[1]-1),
		}
		return [action for action in self.actions if not self.is_dangerous_on_board(candidates[action], board_state)]

	def play_board_state(self, board_state):
		my_snake = self.get_my_snake(board_state)

		if not my_snake.alive:
			if self.last_state is not None:
				self.update_q_table(self.last_state, self.last_action, -100, None)
				self.last_state = None
				self.save_q_table()
			return 'N'

		current_state = self.get_state_from_board(board_state)
		current_score = my_snake.score

		if self.last_state is not None and self.training_enabled:
			reward = -0.1
			if current_score > self.last_score:
				reward = 20
			self.update_q_table(self.last_state, self.last_action, reward, current_state)

		safe_actions = self.get_safe_actions_from_board(board_state)
		if random.random() < self.epsilon:
			action = random.choice(safe_actions if safe_actions else self.actions)
		else:
			q_values = self.get_q_values(current_state)
			if safe_actions:
				safe_q = {a: q_values[a] for a in safe_actions}
				action = max(safe_q, key=safe_q.get)
			else:
				action = max(q_values, key=q_values.get)

		self.last_state = current_state
		self.last_action = action
		self.last_score = current_score

		if board_state.turn % 50 == 0:
			self.save_q_table()

		return action

	def get_state(self, my_snake):
		return self.get_state_from_board(self.board_state_from_game())

	def is_dangerous(self, pos):
		return self.is_dangerous_on_board(pos, self.board_state_from_game())

	def find_goal(self, my_snake):
		return self.find_goal_on_board(self.board_state_from_game())

	def get_q_values(self, state):
		if state not in self.q_table:
			self.q_table[state] = {a: 0.0 for a in self.actions}
		return self.q_table[state]

	def update_q_table(self, state, action, reward, next_state):
		# Solo actualiza si estamos en modo entrenamiento
		if not self.training_enabled:
			return
		current_q = self.get_q_values(state)[action]
		max_future_q = 0 if next_state is None else max(self.get_q_values(next_state).values())
		new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
		self.q_table[state][action] = new_q

	def end_episode(self):
		"""
		Llamado por el runner al terminar cada episodio.
		Aplica la penalizacion terminal si el agente murio y no la proceso aun.
		"""
		if self.game is not None and not self.game.snakes[self.playerID].isAlive and self.last_state is not None:
			self.update_q_table(self.last_state, self.last_action, -100, None)
			self.last_state = None
			self.last_action = None

	def save_model(self):
		"""
		Alias de save_q_table() para que el runner use una API uniforme
		independientemente de si el agente usa Q-Table o una red neuronal.
		"""
		self.save_q_table()

	def get_safe_actions(self, head):
		return self.get_safe_actions_from_board(self.board_state_from_game())

	def play(self, im):
		if isinstance(im, BoardState):
			return self.play_board_state(im)
		if self.game is not None and not self.game.snakes[self.playerID].isAlive:
			if self.last_state is not None:
				self.update_q_table(self.last_state, self.last_action, -100, None)
				self.last_state = None
				self.save_q_table()
			return 'N'
		return self.play_board_state(self.board_state_from_game())
