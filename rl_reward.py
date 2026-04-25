"""
Reward function design for the Snake RL agent.
This module provides a configurable reward function to train the first headless RL agent.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from board_state import BoardState, SnakeState


@dataclass(frozen=True)
class RewardConfig:
    """
    Configuration parameters for the RL reward function.
    Adjusting these weights will change the agent's behavior priorities.
    """
    # 1. Eating fruit
    fruit_multiplier: float = 1.0       # Multiplier for points gained from eating fruit (e.g. fruit values are 10, 15, 20)
    
    # 2. Surviving
    survival_reward: float = 0.01       # Small positive reward for surviving a turn (dense reward)
    
    # 3. Kills and combat
    kill_multiplier: float = 2.0        # Multiplier for points gained from a valid kill (kill score is 30)
    
    # 4. Death penalties
    death_penalty: float = -50.0        # Base penalty for dying
    silly_death_penalty: float = -30.0  # Additional penalty for obvious mistakes (walls, own body)
    
    # 5. Winning the game
    win_reward: float = 100.0           # Huge reward for winning the game
    draw_reward: float = 0.0            # Reward for tying
    
    # 6. Strategic constraints
    # Penalize dying while not a hunter (score < 120) or dying to a bigger snake.
    # This teaches the agent to respect the 120 threshold and avoid impossible kills.
    bad_combat_penalty: float = -20.0   


def compute_reward(
    previous_state: BoardState,
    action: str,
    next_state: BoardState,
    controlled_snake_id: int,
    step_info: Dict[str, Any],
    config: RewardConfig = RewardConfig()
) -> float:
    """
    Computes the step reward for a specific snake.

    Args:
        previous_state: The state of the board before the action was taken.
        action: The action taken by the controlled snake ("N", "S", "E", "W").
        next_state: The state of the board after the action was taken.
        controlled_snake_id: The player_id of the snake we are computing the reward for.
        step_info: Additional info dictionary from the environment step.
        config: RewardConfig weights.

    Returns:
        float: The calculated reward for the current step.
    """
    prev_snake = _get_snake(previous_state, controlled_snake_id)
    next_snake = _get_snake(next_state, controlled_snake_id)
    
    # If the snake doesn't exist in either state, no reward can be computed
    if not prev_snake or not next_snake:
        return 0.0 

    reward = 0.0
    
    # 1. Surviving
    if prev_snake.alive and next_snake.alive:
        reward += config.survival_reward
        
    # 2. Eating fruit
    # Fruit score only goes up when eating fruits
    fruit_score_diff = next_snake.fruit_score - prev_snake.fruit_score
    if fruit_score_diff > 0:
        reward += fruit_score_diff * config.fruit_multiplier
        
    # 3. Killing rivals
    # Total score includes fruit score and kill score. We isolate the kill score.
    prev_kill_score = prev_snake.score - prev_snake.fruit_score
    next_kill_score = next_snake.score - next_snake.fruit_score
    kill_score_diff = next_kill_score - prev_kill_score
    
    if kill_score_diff > 0:
        # A valid kill was performed (advantageous and > 120 threshold)
        reward += kill_score_diff * config.kill_multiplier
        
    # 4. Death Penalties
    if prev_snake.alive and not next_snake.alive:
        reward += config.death_penalty
        
        # Analyze the cause of death
        is_silly = _is_silly_death(previous_state, prev_snake, action)
        if is_silly:
            # Died by hitting a wall or obvious suicide (own body)
            reward += config.silly_death_penalty
        else:
            # Died by colliding with another snake.
            # If we were not a hunter (fruit_score < 120), any collision is either a mutual death
            # or we get eaten. Both are bad combat decisions.
            # If we were a hunter, and we died, it means we hit an even bigger hunter. Also a bad combat decision.
            reward += config.bad_combat_penalty
            
    # 5. Winning games
    # If the game just ended in this step
    if not previous_state.terminal_reason and next_state.terminal_reason:
        if next_state.winner_id == controlled_snake_id:
            reward += config.win_reward
        elif next_state.terminal_reason == "draw" and next_snake.alive:
            # We survived until a draw
            reward += config.draw_reward
            
    return reward


def _get_snake(state: BoardState, player_id: int) -> Optional[SnakeState]:
    """Helper to find a snake in a BoardState by player_id."""
    for s in state.snakes:
        if s.player_id == player_id:
            return s
    return None


def _is_silly_death(state: BoardState, prev_snake: SnakeState, action: str) -> bool:
    """
    Approximates if a death was "silly" (e.g. wall collision or self collision).
    This doesn't re-simulate the whole engine but checks the intended next head position.
    """
    head_r, head_c, _ = prev_snake.head
    
    if action == "N":
        head_r -= 1
    elif action == "S":
        head_r += 1
    elif action == "E":
        head_c += 1
    elif action == "W":
        head_c -= 1
    else:
        # If action is invalid, the engine forces the snake to continue in its current direction
        # or it might be considered a silly behavior overall. We assume it continues.
        pass # Depending on engine, it might use prev direction. Let's just use the logic below.

    # 1. Hitting the wall (Out of bounds)
    if head_r < 0 or head_r >= state.rows or head_c < 0 or head_c >= state.cols:
        return True
        
    # 2. Hitting own body (ignoring the tail, which will move away unless we just ate fruit, but this is a good approximation)
    # We check if the next head position is within the snake's current body (excluding the very last segment)
    body_cells = [(r, c) for r, c, _ in prev_snake.body]
    if len(body_cells) > 1:
        # the tail might move, so we exclude the last element to be safe
        if (head_r, head_c) in body_cells[:-1]:
            return True

    return False
