#---------------------------------------
# Quick diagnostic: Does the Congklak game itself favor P1 or P2?
# Runs 500 games of Random vs Random.
#---------------------------------------
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from congklak import Congklak
from player import Random_player
import numpy as np

num_games = 500
p1_wins = 0
p2_wins = 0
draws = 0

for i in range(num_games):
    g = Congklak()
    g.Ini_board()
    rp = Random_player()
    
    while not g.Check_game_end():
        move = rp.action(g)
        g.Play_action(move)
    
    winner = g.Get_winner()
    if winner == 1:
        p1_wins += 1
    elif winner == -1:
        p2_wins += 1
    else:
        draws += 1

print(f"Random vs Random ({num_games} games)")
print(f"P1 wins: {p1_wins} ({p1_wins/num_games*100:.1f}%)")
print(f"P2 wins: {p2_wins} ({p2_wins/num_games*100:.1f}%)")
print(f"Draws:   {draws} ({draws/num_games*100:.1f}%)")

if p2_wins > p1_wins * 1.2:
    print("\n⚠️  The game has a significant P2 advantage!")
elif p1_wins > p2_wins * 1.2:
    print("\n⚠️  The game has a significant P1 advantage!")
else:
    print("\n✅ The game appears roughly balanced.")
