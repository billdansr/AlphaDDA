#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from copy import deepcopy
import math

class Node():
    def __init__(self, board, player, move=None, parent=None):
        self.board = board
        self.player = player
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

class MCTS():
    def __init__(self, game):
        self.game = game
        self.num_sim = 100

    def Run(self):
        # We need a clean game instance for simulation
        root = Node(deepcopy(self.game.board), self.game.current_player)

        for _ in range(self.num_sim):
            node = root
            temp_game = deepcopy(self.game)
            temp_game.board = deepcopy(node.board)
            temp_game.current_player = node.player

            # 1. Selection
            while len(node.children) > 0:
                node = self.Select_child(node)
                temp_game.Play_action(node.move)

            # 2. Expansion
            winner = temp_game.Get_winner()
            is_end = temp_game.Check_game_end()
            
            if not is_end:
                valid_moves = temp_game.Get_valid_moves()
                for move in valid_moves:
                    sim_game = deepcopy(temp_game)
                    sim_game.Play_action(move)
                    child = Node(deepcopy(sim_game.board), sim_game.current_player, move, node)
                    node.children.append(child)
                node = node.children[np.random.randint(len(node.children))]
                temp_game.Play_action(node.move)

            # 3. Simulation (Rollout)
            while not temp_game.Check_game_end():
                moves = temp_game.Get_valid_moves()
                temp_game.Play_action(np.random.choice(moves))
            
            # 4. Backpropagation
            result = temp_game.Get_winner()
            self.Backpropagate(node, result)

        # Output move with most visits
        counts = [child.visits for child in root.children]
        return root.children[np.argmax(counts)].move

    def Select_child(self, node):
        # UCB1
        log_total = math.log(node.visits)
        best_score = -float('inf')
        best_child = None
        
        for child in node.children:
            if child.visits == 0:
                return child
            # Perspective: node.player is the one choosing
            # We want to maximize wins for node.player
            win_rate = child.wins / child.visits
            if child.player != node.player:
                win_rate = 1 - win_rate # Opponent turn result
            
            score = win_rate + 1.41 * math.sqrt(log_total / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def Backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            if result == node.player:
                node.wins += 1
            elif result == 0:
                node.wins += 0.5
            node = node.parent
