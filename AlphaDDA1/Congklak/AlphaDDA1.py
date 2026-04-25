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
    def __init__(self, game, net = None, params = Parameters(), num_mean = 5, N_MAX = 800, states = None):
        self.num_moves = None
        self.params = params
        
        # DDA parameters
        self.max_num_values = num_mean
        self.estimated_outcome_queue = []
        self.N_MAX = N_MAX
        self.N_MIN = 60 # Minimum simulations to maintain basic tactical awareness
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

        for m in valid_actions:
            temp_g.board = deepcopy(node.board)
            temp_g.current_player = node.player
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
        As defined in PeerJ-CS 1123 (Fujita, 2022).
        """
        # 1. Get the current estimated value (v) from the NN
        # States represent the game history; we use the latest state.
        temp_g = Congklak()
        temp_g.board = deepcopy(self.root.board)
        temp_g.current_player = self.root.player
        temp_g.seq_boards.buf = deepcopy(self.root.history)
        
        _, v = self.nn.predict(temp_g.Get_states())
        
        # 2. Update the rolling window of outcomes
        self.estimated_outcome_queue.append(float(v))
        if len(self.estimated_outcome_queue) > self.max_num_values:
            self.estimated_outcome_queue.pop(0)
            
        # 3. Calculate average win_score (relative to AI player)
        win_score = mean(self.estimated_outcome_queue)
        
        # 4. Simulation Adjustment Logic:
        # If AI is winning (win_score > 0), reduce simulations to give the human a chance.
        # If AI is losing or even (win_score <= 0), use N_MAX.
        
        if win_score > 0:
            # Linear reduction: N_sim = N_MAX * (1 - win_score)
            # We clip it to N_MIN so it doesn't become completely random.
            reduction_factor = 1.0 - (win_score ** 2) # Using squared for smoother drop
            new_sims = int(self.N_MAX * reduction_factor)
            self.params.num_mcts_sims = max(self.N_MIN, new_sims)
        else:
            self.params.num_mcts_sims = self.N_MAX
            
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
                
                psa_vector, v = self.nn.predict(temp_g.Get_states())
                valid_moves = temp_g.Get_valid_moves()
                
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
