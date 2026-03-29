from train_mp import Train
from parameters import Parameters
import torch

def test_training():
    params = Parameters()
    # Override for fast test
    params.num_mcts_sims = 5
    params.num_games = 2
    params.num_processes_training = 1
    params.num_iterations = 1
    params.batch_size = 32
    params.epochs = 1
    params.devices = ["cpu"] # Use CPU for quick test
    
    tr = Train()
    tr.params = params
    tr.net.params = params
    tr.net.device = "cpu"
    
    print("Starting minimal training run...")
    tr.Run()
    print("Minimal training run SUCCESSFUL")

if __name__ == "__main__":
    test_training()
