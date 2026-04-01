#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
from copy import deepcopy
import random
from congklak import Congklak
from parameters import Parameters

class Node():
    def __init__(self, state, player, move = None, terminal = False, winner = 0, parent = None):
        self.p        = player
        self.move     = move
        self.state    = state
        self.children = []
        self.parent   = parent
        self.terminal = terminal
        self.winner   = winner
        self.value    = None

    def Get_state(self):
        return deepcopy(self.state)

    def Get_player(self):
        return deepcopy(self.p)

    def Add_child(self, state, player, move, terminal, winner):
        child = Node(state, player, move, terminal, winner, self)
        self.children.append(child)

class Minimax:
    def __init__(self, game):
        self.g = game
        self.p = self.g.current_player
        self.params = Parameters()

        # In case the board passed isn't raw but a wrapped object, we extract the board
        board_data = self.g.Get_board() if hasattr(self.g, 'Get_board') else self.g.board
        
        self.root = Node(state = board_data, player = self.g.current_player)

        # Depth 3 is a good balance for Congklak's branching factor
        self.depth = 3 

    def Expand_node(self, node):
        temp_g = Congklak()
        temp_g.board = node.Get_state()
        temp_g.current_player = node.Get_player()
        valid_moves = temp_g.Get_valid_moves()
        for m in valid_moves:
            # We reset a temp board state to simulate the move
            temp_g.board = node.Get_state()
            temp_g.current_player = node.Get_player()
            # Note: Play_action in new Congklak takes relative move (0-6)
            # Need to verify if the node uses relative or absolute index
            temp_g.Play_action(m)
            player = temp_g.current_player
            terminal = temp_g.Check_game_end()
            winner = temp_g.Get_winner()
            state = temp_g.Get_board()
            node.Add_child(state, player, m, terminal, winner)

    def Make_tree(self, node, depth):
        depth -= 1
        if node.terminal != True and depth >= 0:
            self.Expand_node(node)
            for i in node.children:
                self.Make_tree(i, depth)

    def Run(self):
        node = self.root
        self.Make_tree(node, self.depth)
        
        if not self.root.children:
            # Fallback to random if no expansion possible
            valid_moves = self.g.Get_valid_moves()
            return random.choice(valid_moves) if len(valid_moves) > 0 else 0
            
        best_node = self.Search(self.root)
        return best_node.move

    def Evaluate(self, s, p):
        # Congklak evaluation: Difference in store seeds
        # New Store indices: P1=7, P2=15
        score_p1 = s[7]
        score_p2 = s[15]
        
        # v is normalized to 'p' perspective
        if p == 1:
            v = score_p1 - score_p2
        else:
            v = score_p2 - score_p1
            
        return v

    def Search(self, root):
        stack = [root] 

        while len(stack) != 0:
            n = stack.pop()

            if len(n.children) != 0:
                stack += n.children
            else:
                v = 0
                if n.terminal:
                    # High value for win, low for loss
                    # n.winner is the actual winner (1 or -1)
                    # root.p is the identity of the searching player
                    if n.winner == root.p:
                        v = 1000000
                    elif n.winner == 0:
                        v = 0
                    else:
                        v = -1000000
                else:
                    if len(n.children) == 0:
                        v = self.Evaluate(n.state, root.p)
                    else:
                        v = n.value

                n.value = v

                # Propagate value up the tree
                curr = n
                while curr.parent != None:
                    curr = curr.parent

                    if curr.value == None:
                        curr.value = v
                    elif curr.p == root.p and curr.value < v:
                        # Maximize for the root player
                        curr.value = v
                    elif curr.p != root.p and curr.value > v:
                        # Minimize for the opponent player
                        curr.value = v
        
        # After propagation, pick the best child of root
        best_val = -float('inf')
        best_child = root.children[0]
        for child in root.children:
            if child.value is not None and child.value > best_val:
                best_val = child.value
                best_child = child
                
        return best_child
