from congklak import Congklak
import numpy as np

g = Congklak()
states = g.Get_states()

print("Shape of states:", states.shape)
print("Type of states:", type(states))
print("First frame, Channel 1 (Current Player):\n", states[0])
print("First frame, Channel 2 (Opponent):\n", states[1])
print("Perspective Channel (Constant):\n", states[2])

assert states.shape == (3, 2, 8), "Error: Shape mismatch!"
print("\n✅ Shape and output look perfect for AlphaZero training!")
