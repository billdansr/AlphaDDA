from congklak import Congklak

def test_game():
    g = Congklak()
    g.Print_board()
    
    # Simulate a few turns
    print("Player 1 moves hole 0")
    g.Play_action(0)
    g.Print_board()
    
    print("Valid moves for next player:", g.Get_valid_moves())
    
    # Test capturing if possible or just more turns
    # We can manually set the board to test capture
    g.board[:] = 0
    g.board[0] = 1 # P1 hole 0
    g.board[14] = 10 # P2 hole 14 (opposite of 0)
    g.current_player = 1
    print("\nManually testing capture at hole 0:")
    g.Print_board()
    g.Play_action(0) # Drops 1 in hole 1 (empty). 
    # Wait, if hole 1 is empty, it should capture hole 13.
    # Let's try: drop in hole 1, opposite is hole 13.
    g.board[:] = 0
    g.board[0] = 1
    g.board[13] = 5
    g.current_player = 1
    print("\nTesting capture: P1 drops last shell in empty hole 1. Opposite hole 13 has 5 shells.")
    g.Print_board()
    g.Play_action(0)
    g.Print_board()
    # Expectation: P1 store should have 5 + 1 = 6 shells.
    if g.board[7] == 6:
        print("Capture SUCCESS")
    else:
        print(f"Capture FAILED: P1 store has {g.board[7]}")

    # Test extra turn (dropping in store)
    g.board[:] = 0
    g.board[0] = 7 # P1 hole 0 has 7 shells. Store is at 7.
    g.current_player = 1
    print("\nTesting Extra Turn: P1 drops last shell in store (7th hole from 0).")
    g.Play_action(0)
    g.Print_board()
    if g.current_player == 1:
        print("Extra Turn SUCCESS")
    else:
        print("Extra Turn FAILED")

if __name__ == "__main__":
    test_game()
