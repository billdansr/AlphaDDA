#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import torch
import multiprocessing

class Parameters:
    def __init__(self):
        # Game settings for Congklak
        self.board_x     = 2            
        self.board_y     = 8            
        self.action_size = 7            
        self.p1          = 1            
        self.p2          = -1           
        self.initial_shells = 7         

        #------------------------
        # AlphaZero Parameters
        is_cuda = torch.cuda.is_available()
        self.num_processes_training = 10 if is_cuda else max(1, multiprocessing.cpu_count() - 1)
        self.num_processes_test = 10 if is_cuda else max(1, multiprocessing.cpu_count() - 1)
        
        device_str = "cuda:0" if is_cuda else "cpu"
        self.devices = [device_str] * self.num_processes_training
        
        # Learning defaults
        self.num_iterations      = 600
        self.num_games           = 30  
        self.checkpoint_interval = 5
        self.num_test            = 10 

        # MCTS defaults
        self.num_mcts_sims = 80                
        self.cpuct         = 1.25               
        self.opening_train = 4                  
        self.opening_test  = 0                  
        self.opening       = self.opening_train 
        self.Temp          = 50                 
        self.rnd_rate      = 0.2                

        # Neural Network architecture
        self.input_size     = 20000                   
        self.k_boards       = 1                       
        self.input_channels = (self.k_boards * 2) + 1 
        self.num_filters    = 256                     
        self.num_filters_p  = 2                       
        self.num_filters_v  = 1                       
        self.num_res        = 3                       
        self.epochs         = 1                       
        self.batch_size     = 2048                    
        self.lam            = 2e-1                    
        self.weight_decay   = 1e-4                    
        self.momentum       = 0.9                     
