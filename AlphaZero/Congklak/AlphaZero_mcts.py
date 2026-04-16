#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from copy import deepcopy
import random
import math
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
    def __init__(self, game, net = None, params = Parameters(), is_training = False):
        self.num_moves = None
        self.params = params
        self.is_training = is_training
        if net == None:
            self.nn = nnet(params=params)
        else:
            self.nn = net

        # Make the root node.
        self.root = Node(board = game.Get_board(), history = deepcopy(game.seq_boards.buf), player = game.current_player)

    def softmax(self, x):
        x = np.exp(x / self.params.Temp)
        return x/np.sum(x)

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

    def Run(self):
        # Use sims_test if not training
        sims = self.params.num_mcts_sims if self.is_training else self.params.num_mcts_sims_test
        for _ in range(sims):
            node = self.root
            while len(node.children) != 0:
                node = self.Search(node)

            v = 0
            if node.terminal:
                # Terminal value from node.player perspective
                # node.winner is absolute (1 or -1). 
                v = node.winner * node.player
            else:
                # Need to reconstruct game to get NN input
                temp_g = Congklak()
                temp_g.board = node.board
                temp_g.current_player = node.player
                temp_g.seq_boards.buf = deepcopy(node.history)
                
                psa_vector, v = self.nn.predict(temp_g.Get_states())
                
                valid_moves = temp_g.Get_valid_moves()
                
                # Masking and normalizing psa
                mask = np.zeros(self.params.action_size)
                mask[valid_moves] = 1
                psa_vector = psa_vector * mask
                sum_psa = np.sum(psa_vector)
                if sum_psa > 0:
                    psa_vector /= sum_psa
                else:
                    psa_vector = mask / np.sum(mask)

                # Dirichlet Noise only during training at root node
                if self.is_training and node == self.root and getattr(self.params, 'dirichlet_alpha', None) is not None:
                    noise = np.random.dirichlet([self.params.dirichlet_alpha] * len(valid_moves))
                    noise_full = np.zeros(self.params.action_size)
                    noise_full[valid_moves] = noise
                    psa_vector = (1 - self.params.dirichlet_eps) * psa_vector + self.params.dirichlet_eps * noise_full
                    
                    # Renormalize to ensure sum=1
                    psa_vector /= np.sum(psa_vector)

                self.Expand_node(node, psa_vector)

            self.Back_prop(node, v)

        return self.Decide_move()

    def Decide_move(self):
        visits = np.array([i.nsa for i in self.root.children])
        if self.num_moves > self.params.opening:
            return self.root.children[np.argmax(visits)].move
        else:
            pi = self.softmax(visits)
            best_child = self.root.children[np.random.choice(len(self.root.children), p = pi.tolist())]
            return best_child.move

    def Search(self, node):
        N = np.sum(np.array([i.nsa for i in node.children]))
        # AlphaZero exclusively uses PUCT (ucb-like formula) combined with Dirichlet noise at the root.
        # Random epsilon-greedy is removed.
        best_child = node.children[np.argmax(np.array([self.l(i.qsa, i.nsa, i.psa, N) for i in node.children]))]
        return best_child

    def l(self, qsa, nsa, psa, N):
        return qsa + self.params.cpuct * psa * math.sqrt(N) / (nsa + 1)

    def Back_prop(self, node, v):
        """
        Backpropagate the value 'v' up the tree.
        'v' is returned by the network from the perspective of node.player (the player in the leaf state).
        """
        # v is from the perspective of the leaf player
        while node != self.root:
            node.nsa += 1
            
            # W and Q are from the perspective of the parent player.
            # If parent and child are the same player (extra turn), use v directly.
            # If they are different players, parent's value is the negative of the child's value.
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
