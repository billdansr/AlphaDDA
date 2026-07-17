#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from copy import deepcopy
import math
from congklak import Congklak

class Node():
    def __init__(self, state, player, move = None, terminal = False, winner = 0, parent = None):
        self.n        = 0 # the visit count
        self.q        = 0
        self.p        = player
        self.move     = move
        self.state    = state
        self.children = []
        self.parent   = parent
        self.terminal = terminal
        self.winner   = winner

    def Get_state(self):
        return deepcopy(self.state)

    def Get_player(self):
        return deepcopy(self.p)

    def Add_child(self, state, player, move, terminal, winner):
        child = Node(state, player, move, terminal, winner, self)
        self.children.append(child)

class MCTS:
    def __init__(self, game):
        self.g = game
        self.p = self.g.current_player

        self.root = Node(state = self.g.Get_board(), player = self.g.current_player)

        self.num_sim = 100 # the number of simulations
        self.th_open_leaf = 5 # A leaf node is expanded if its visit count is >= this number.

    def Expand_node(self, node):
        temp_g = Congklak()
        temp_g.board = node.Get_state()
        temp_g.current_player = node.Get_player()
        valid_moves = temp_g.Get_valid_moves()
        
        # If no valid moves but game is not ended (skipped turn)
        if len(valid_moves) == 0 and not temp_g.Check_game_end():
            temp_g.current_player *= -1
            valid_moves = temp_g.Get_valid_moves()

        for m in valid_moves:
            temp_g.board = node.Get_state()
            temp_g.current_player = node.Get_player()
            temp_g.Play_action(m)
            player = temp_g.current_player
            terminal = temp_g.Check_game_end()
            winner = temp_g.Get_winner()
            state = temp_g.Get_board()
            node.Add_child(state, player, m, terminal, winner)

        if len(node.children) > 0:
            node.children[0].n = 1

    def Run(self):
        for _ in range(self.num_sim):
            node = self.root
            while node.terminal == False:
                if len(node.children) == 0 and (node == self.root or node.n >= self.th_open_leaf):
                    self.Expand_node(node)
                else:
                    if len(node.children) == 0:
                        # Playout step
                        node.winner = self.random_play(node)
                        break
                    else:
                        # Selection step
                        node = self.Search(node)

            reward = deepcopy(node.winner)
            self.BACKUP(node, reward)

        return self.Decide_move()

    def random_play(self, node):
        temp_g = Congklak()
        temp_g.board = node.Get_state()
        temp_g.current_player = node.Get_player()

        while not temp_g.Check_game_end():
            valid_moves = temp_g.Get_valid_moves()
            if len(valid_moves) == 0:
                temp_g.current_player *= -1
                continue
            move = np.random.choice(valid_moves)
            temp_g.Play_action(move)
            
        return temp_g.Get_winner()

    def l(self, q, n, N):
        # UCT score
        return float(q)/(n + 1e-7) + 0.5 * math.sqrt(2 * math.log(N + 1) / (n + 1e-7))

    def Search(self, node):
        N = np.sum(np.array([i.n for i in node.children])) # the visit count of the parent node
        best_child = node.children[np.argmax(np.array([self.l(i.q, i.n, N) for i in node.children]))]
        return best_child

    def Decide_move(self):
        # The move corresponding to the child node with the maximum visit count is selected.
        return self.root.children[np.argmax(np.array([i.n for i in self.root.children]))].move

    def BACKUP(self, node, reward):
        while node != None:
            node.n += 1
            if node.parent is not None:
                # node.parent.p is the player who chose the move leading to node.
                # So we update from their perspective.
                node.q += reward * node.parent.p
            node = node.parent
