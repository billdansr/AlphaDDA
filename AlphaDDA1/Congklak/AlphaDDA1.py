#---------------------------------------
# AlphaDDA1 Implementation for Congklak
# Based on: Fujita (2022) - PeerJ Computer Science
# Purpose: Research Evaluation for Thesis
#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from copy import deepcopy
import random
import math
from statistics import mean

from nn import NNetWrapper as nnet
from parameters import Parameters
from congklak import Congklak

class Node():
    def __init__(self, board, history, player, move = None, psa = 0, terminal = False, winner = 0, parent = None):
        self.nsa      = 0
        self.wsa      = 0
        self.qsa      = 0
        self.psa      = psa
        self.player   = player # who IS moving in this state
        self.move     = move
        self.board    = board
        self.history  = history # contents of seq_boards.buf
        self.children = []
        self.parent   = parent
        self.terminal = terminal
        self.winner   = winner

    def Add_child(self, board, history, player, move, psa, terminal, winner, parent):
        child = Node(board = board, history = history, player = player, move = move, psa = psa, terminal = terminal, winner = winner, parent = parent)
        self.children.append(child)

class A_MCTS:
    def __init__(self, game, net = None, params = Parameters(), num_mean = 1, X0 = 0.0, A = 10.0, N_MAX = 400, states = None):
        self.num_moves = None
        self.params = params
        
        # DDA parameters — faithfully from PeerJ-CS 1123 (Fujita, 2022)
        self.max_num_values = num_mean
        self.estimated_outcome_queue = []
        self.A = A
        self.X0 = X0
        self.N_MAX = N_MAX
        self.states_history = states if states is not None else []
        
        if net == None:
            self.nn = nnet(params=params)
        else:
            self.nn = net

        # Make the root node.
        self.root = Node(board = game.Get_board(), history = deepcopy(game.seq_boards.buf), player = game.current_player)

    def softmax(self, x):
        # Numerically stable softmax
        x = x.astype(float)
        x = (x - np.max(x)) / self.params.Temp
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def Expand_node(self, node, psa_vector):
        temp_g = Congklak()
        temp_g.board = deepcopy(node.board)
        temp_g.current_player = node.player
        valid_actions = temp_g.Get_valid_moves()
        
        moving_player = node.player
        if len(valid_actions) == 0 and not temp_g.Check_game_end():
            temp_g.current_player *= -1
            valid_actions = temp_g.Get_valid_moves()
            moving_player = temp_g.current_player

        for m in valid_actions:
            temp_g.board = deepcopy(node.board)
            temp_g.current_player = moving_player
            temp_g.seq_boards.buf = deepcopy(node.history) 
            
            temp_g.Play_action(m)
            
            psa = psa_vector[m]
            node.Add_child(
                board = temp_g.Get_board(),
                history = deepcopy(temp_g.seq_boards.buf),
                player = temp_g.current_player,
                move = m,
                psa = psa,
                terminal = temp_g.Check_game_end(),
                winner = temp_g.Get_winner(),
                parent = node
            )

    def Update_DDA_Simulations(self):
        """
        Implements AlphaDDA1: Adjusting playing strength based on the predicted game outcome.
        Formula: N_sim = 10^(-A * (avg_win_score * player + X0))
        Directly from PeerJ-CS 1123 (Fujita, 2022), matching the Connect4 reference implementation.
        """
        # 1. Get the current estimated value (v) from the NN
        temp_g = Congklak()
        temp_g.board = deepcopy(self.root.board)
        temp_g.current_player = self.root.player
        temp_g.seq_boards.buf = deepcopy(self.root.history)
        
        _, v = self.nn.predict(temp_g.Get_states())
        
        # 2. Update the rolling window of outcomes
        self.estimated_outcome_queue.append(float(v))
        if len(self.estimated_outcome_queue) > self.max_num_values:
            self.estimated_outcome_queue.pop(0)
            
        # 3. Calculate average win_score relative to the current AI player
        # Note: v is already canonical (relative to current player), so we don't multiply by player side.
        win_score = mean(self.estimated_outcome_queue)
        
        # 4. Paper's exact formula: N_sim = ceil(10^(-A * (win_score + X0)))
        # Numerical Safety: Clip the exponent to prevent OverflowError
        exponent = -self.A * (win_score + self.X0)
        exponent = min(exponent, 10) # 10^10 is plenty large enough before clipping to N_MAX
        
        new_sims = math.ceil(10 ** exponent)
        
        # 5. Clip to [1, N_MAX]
        self.params.num_mcts_sims = max(1, min(new_sims, self.N_MAX))
            
        print(f"AlphaDDA1: v={v:.3f}, avg_v={win_score:.3f} -> Sims: {self.params.num_mcts_sims}")

    def Run(self):
        # Update simulations before starting MCTS
        self.Update_DDA_Simulations()

        for _ in range(self.params.num_mcts_sims):
            node = self.root
            while len(node.children) != 0:
                node = self.Search(node)

            v = 0
            if node.terminal:
                # Absolute winner flip perspective: v = winner * node.player
                # This ensures v is always relative to the player at the leaf.
                v = node.winner * node.player
            else:
                temp_g = Congklak()
                temp_g.board = deepcopy(node.board)
                temp_g.current_player = node.player
                temp_g.seq_boards.buf = deepcopy(node.history)
                
                valid_moves = temp_g.Get_valid_moves()
                if len(valid_moves) == 0 and not temp_g.Check_game_end():
                    temp_g.current_player *= -1
                    valid_moves = temp_g.Get_valid_moves()
                
                psa_vector, v = self.nn.predict(temp_g.Get_states())
                
                mask = np.zeros(self.params.action_size)
                mask[valid_moves] = 1
                psa_vector = psa_vector * mask
                sum_psa = np.sum(psa_vector)
                if sum_psa > 0:
                    psa_vector /= sum_psa
                else:
                    psa_vector = mask / np.sum(mask)

                self.Expand_node(node, psa_vector)

            self.Back_prop(node, v)

        return self.Decide_move()

    def Decide_move(self):
        visits = np.array([i.nsa for i in self.root.children])
        if self.num_moves is not None and self.num_moves > self.params.opening:
            return self.root.children[np.argmax(visits)].move
        else:
            pi = self.softmax(visits)
            best_child = self.root.children[np.random.choice(len(self.root.children), p = pi.tolist())]
            return best_child.move

    def Search(self, node):
        N = np.sum(np.array([i.nsa for i in node.children]))
        # PUCT search
        best_child = node.children[np.argmax(np.array([self.l(i.qsa, i.nsa, i.psa, N) for i in node.children]))]
        return best_child

    def l(self, qsa, nsa, psa, N):
        return qsa + self.params.cpuct * psa * math.sqrt(N) / (nsa + 1)

    def Back_prop(self, node, v):
        while node != self.root:
            node.nsa += 1
            # Flip perspective if parent is a different player
            if node.parent.player != node.player:
                v = -v
            node.wsa += v
            node.qsa = node.wsa / node.nsa
            node = node.parent

    def Get_prob(self):
        prob = np.zeros(self.params.action_size)
        for i in self.root.children:
            prob[i.move] += i.nsa
        prob /= np.sum(prob)
        return prob
