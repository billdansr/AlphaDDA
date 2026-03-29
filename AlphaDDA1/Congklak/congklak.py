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
                self.winner = 0.0001 # Draw (almost 0 but distinct?) Wait, let's keep winner = 0 for tie, but since Connect4 uses 0 for non-end, we have to carefully distinguish. AlphaZero handles ties by value = 0.
            
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
        # Return a canonical state from the POV of current_player
        # Connect4 was shape (channels, x, y) -> (3, x, y)
        b = self.board
        
        state1 = np.zeros(16)
        state2 = np.zeros(16)
        
        # We'll normalize shell counts. They can reach up to 98.
        # But for CNN we can just use raw integer values or normalize by 98.0
        # Wait, AlphaZero Connect4 uses exactly 1s and -1s. Here we can use shell counts / 98.0.
        norm_b = b / 98.0 # Total shells is 98
        
        if self.current_player == self.params.p1:
            state1 = norm_b.copy() # Our shells
            # Let's frame it as state1 = our view, state2 = we mirror the board exactly?
            # ACTUALLY, canonical format means it always looks like P1 is playing.
            canonical_board = norm_b.copy()
            player_indicator = np.ones((self.params.board_x, self.params.board_y))
        else:
            # P2 to move: swap P1 and P2 halves so holes 0-7 are the current player's
            canonical_board = np.zeros(16)
            canonical_board[0:8] = norm_b[8:16]
            canonical_board[8:16] = norm_b[0:8]
            player_indicator = np.zeros((self.params.board_x, self.params.board_y))
            
        # The NN needs k_boards * 2 + 1 channels.
        # k_boards = 1 -> 3 channels.
        # In connect4 they did:
        # channel 1: 1 where P_1 has stones
        # channel 2: 1 where P_2 has stones
        # channel 3: turn indicator (all 1 for P1, all 0 for P2)
        # We can do:
        # channel 1: Current player's shell counts
        # channel 2: Opponent's shell counts
        # channel 3: Current player turn flag (from seq_boards logic if k_boards > 1) 
        
        temp_states = self.seq_boards.Get_buffer()
        states = []
        for i in range(self.params.k_boards):
            past_b = temp_states[i] / 98.0
            if self.current_player == self.params.p1:
                states.append(past_b[0:8])
                states.append(past_b[8:16])
            else:
                states.append(past_b[8:16])
                states.append(past_b[0:8])

        if self.current_player == 1:
            states.append(np.ones(8))
        else:
            states.append(np.zeros(8))
            
        # Reshape to input_channels x xnum x ynum
        # But states is shape: we have 3 lists of length 8. That's (3, 8). 
        # But board is board_x = 2, board_y = 8.
        # So we should reshape into (3, 2, 8)? No, states append is flat.
        
        # Let's fix this.
        # For each frame (k_boards):
        # We want channel i: our view (2 rows, 8 cols)
        states = []
        for i in range(self.params.k_boards):
            past_b = temp_states[i] / 98.0
            
            p1_side = past_b[0:8]
            p2_side = past_b[8:16]
            
            if self.current_player == self.params.p1:
                channel1 = np.vstack((p1_side, p2_side))
                channel2 = np.vstack((p2_side, p1_side)) # Or something similar. For now let's just make channel 1 = our side of shells, channel 2 = op side.
                states.append(channel1)
                states.append(channel2)
            else:
                channel1 = np.vstack((p2_side, p1_side))
                channel2 = np.vstack((p1_side, p2_side))
                states.append(channel1)
                states.append(channel2)
                
        if self.current_player == self.params.p1:
            states.append(np.ones((2, 8)))
        else:
            states.append(np.zeros((2, 8)))
            
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
                
            # Drop 1
            b[curr_hole] += 1
            shells -= 1
            
            # If last shell drops
            if shells == 0:
                # 1. Drops in own store -> extra turn
                if (self.current_player == self.params.p1 and self._is_p1_store(curr_hole)) or \
                   (self.current_player == self.params.p2 and self._is_p2_store(curr_hole)):
                    extra_turn = True
                    break
                
                # 2. Drops in an empty hole (which got 1 right now)
                elif b[curr_hole] == 1:
                    # Is it our side?
                    if self._is_own_hole(curr_hole, self.current_player):
                        opposite = self._get_opposite_hole(curr_hole)
                        if b[opposite] > 0:
                            # Capture
                            captured = b[opposite] + 1
                            b[opposite] = 0
                            b[curr_hole] = 0
                            
                            store_idx = 7 if self.current_player == self.params.p1 else 15
                            b[store_idx] += captured
                    break
                    
                # 3. Drops in a non-empty hole -> pickup and sow
                elif b[curr_hole] > 1:
                    shells = b[curr_hole]
                    b[curr_hole] = 0
                    
        if not extra_turn:
            self.current_player *= -1
            
        # Optional: handling game end where we need to sweep is done in Check_game_end
        self.seq_boards.add(deepcopy(self.board))
        
if __name__ == '__main__':
    g = Congklak()
    g.Print_board()
    
    while not g.Check_game_end():
        print("Valid moves:", g.Get_valid_moves())
        action = int(input("Move: "))
        g.Play_action(action)
        g.Print_board()
        
    print("Winner:", g.Get_winner())
