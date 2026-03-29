#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from copy import deepcopy
from parameters import Parameters
from ringbuffer import RingBuffer

class Congklak():
    def __init__(self):
        self.params = Parameters()
        self.board_size = 16 # total holes: 0-6 P1 holes, 7 P1 store, 8-14 P2 holes, 15 P2 store
        self.action_size = self.params.action_size
        self.winner = 0
        self.put_location = 0

        self.Ini_board()

    def Ini_board(self):
        self.board = self.Create_board()
        self.current_player = self.params.p1
        self.seq_boards = RingBuffer(self.params.k_boards)
        for i in range(self.params.k_boards):
            self.seq_boards.add(np.zeros(16))
        self.seq_boards.add(deepcopy(self.board))

    def Create_board(self):
        board = np.zeros(self.board_size, dtype=int)
        for i in range(7):
            board[i] = self.params.initial_shells
            board[i+8] = self.params.initial_shells
        return board

    def Print_board(self):
        """
        Prints the board according to Indonesian Congklak standards:
        - Sowing is clockwise (to the left).
        - Store house is on the left-hand side of the player.
        
        P2 holes:  8  9 10 11 12 13 14
        P1 home: [ 7]                  P2 home: [15]
        P1 holes:  6  5  4  3  2  1  0
        """
        b = self.board
        print(f"      P2: {b[8]:2d} {b[9]:2d} {b[10]:2d} {b[11]:2d} {b[12]:2d} {b[13]:2d} {b[14]:2d}")
        print(f"[{b[7]:3d}]                              [{b[15]:3d}]")
        print(f"      P1: {b[6]:2d} {b[5]:2d} {b[4]:2d} {b[3]:2d} {b[2]:2d} {b[1]:2d} {b[0]:2d}")
        print(f"Current Player: {'P1' if self.current_player == self.params.p1 else 'P2'}")
        print("---------------------------------")

    def Get_player(self):
        return deepcopy(self.current_player)

    def Get_board(self):
        return deepcopy(self.board)

    def Get_board_size(self):
        return (self.params.board_x, self.params.board_y)

    def Get_action_size(self):
        return self.action_size
        
    def _is_p1_store(self, idx):
        return idx == 7
        
    def _is_p2_store(self, idx):
        return idx == 15
        
    def _get_opposite_hole(self, idx):
        if 0 <= idx <= 6:
            return 14 - idx
        elif 8 <= idx <= 14:
            return 14 - idx
        return -1
        
    def _is_own_hole(self, idx, player):
        if player == self.params.p1:
            return 0 <= idx <= 6
        else:
            return 8 <= idx <= 14

    def Get_valid_moves(self):
        # A valid move is any own hole with >0 shells
        b = self.board
        moves = []
        if self.current_player == self.params.p1:
            for i in range(7):
                if b[i] > 0:
                    moves.append(i)
        else:
            for i in range(7):
                if b[i+8] > 0:
                    moves.append(i) # return relative action (0-6)
        
        return np.array(moves)

    def Check_game_end(self):
        b = self.board
        
        # Check if P1 side is empty
        p1_empty = np.all(b[0:7] == 0)
        # Check if P2 side is empty
        p2_empty = np.all(b[8:15] == 0)
        
        if p1_empty or p2_empty:
            # Game finishes: sweep remaining shells
            if p1_empty:
                b[15] += np.sum(b[8:15])
                b[8:15] = 0
            if p2_empty:
                b[7] += np.sum(b[0:7])
                b[0:7] = 0
                
            if b[7] > b[15]:
                self.winner = self.params.p1
            elif b[15] > b[7]:
                self.winner = self.params.p2
            else:
                self.winner = 0
            
            return True
        return False
        
    def Get_winner(self):
        if self.board[7] > self.board[15]:
            return self.params.p1
        elif self.board[15] > self.board[7]:
            return self.params.p2
        else:
            # Draw
            return 0

    def Get_states(self):
        """
        Return a canonical state from the POV of current_player.
        Shape: (input_channels, board_x, board_y) -> (3, 2, 8)
        """
        temp_states = self.seq_boards.Get_buffer()
        states = []
        
        # Generate canonical representations for current and history frames
        for i in range(self.params.k_boards):
            past_b = temp_states[i] / 98.0
            
            # Canonicalize: Current player's own side is always at Row 0
            if self.current_player == self.params.p1:
                my_side = past_b[0:8]
                op_side = past_b[8:16]
            else:
                my_side = past_b[8:16]
                op_side = past_b[0:8]
                
            # Channel for current player's shells
            ch1 = np.zeros((2, 8))
            ch1[0] = my_side
            
            # Channel for opponent's shells
            ch2 = np.zeros((2, 8))
            ch2[1] = op_side
            
            states.append(ch1)
            states.append(ch2)
                
        # Perspective channel: Always 1 because the representation is canonical
        states.append(np.ones((2, 8)))
            
        return np.array(states)

    def Play_action(self, action):
        """
        action is an integer 0..6 relative to the current player's side.
        """
        if self.current_player == self.params.p1:
            start_hole = action
        else:
            start_hole = action + 8
            
        b = self.board
        shells = b[start_hole]
        b[start_hole] = 0
        curr_hole = start_hole
        
        extra_turn = False
        
        while shells > 0:
            curr_hole = (curr_hole + 1) % 16
            
            # Skip opponent's store
            if self.current_player == self.params.p1 and self._is_p2_store(curr_hole):
                continue
            if self.current_player == self.params.p2 and self._is_p1_store(curr_hole):
                continue
                
            # Drop 1 shell
            b[curr_hole] += 1
            shells -= 1
            
            # Last shell behavior
            if shells == 0:
                # 1. Drops in own store -> extra turn
                if (self.current_player == self.params.p1 and self._is_p1_store(curr_hole)) or \
                   (self.current_player == self.params.p2 and self._is_p2_store(curr_hole)):
                    extra_turn = True
                    break
                
                # 2. Drops in an empty hole on own side -> capture
                elif b[curr_hole] == 1:
                    if self._is_own_hole(curr_hole, self.current_player):
                        opposite = self._get_opposite_hole(curr_hole)
                        if b[opposite] > 0:
                            captured = b[opposite] + 1
                            b[opposite] = 0
                            b[curr_hole] = 0
                            
                            store_idx = 7 if self.current_player == self.params.p1 else 15
                            b[store_idx] += captured
                    break
                    
                # 3. Drops in a non-empty hole -> pick up and continue sowing
                elif b[curr_hole] > 1:
                    shells = b[curr_hole]
                    b[curr_hole] = 0
                    
        if not extra_turn:
            self.current_player *= -1
            
        self.seq_boards.add(deepcopy(self.board))
        
if __name__ == '__main__':
    g = Congklak()
    g.Print_board()
    
    while not g.Check_game_end():
        print("Valid moves:", g.Get_valid_moves())
        try:
            action = int(input("Move (0-6): "))
            if action in g.Get_valid_moves():
                g.Play_action(action)
                g.Print_board()
            else:
                print("Invalid move, try again.")
        except ValueError:
            print("Please enter a number.")
        
    print("Winner:", g.Get_winner())
