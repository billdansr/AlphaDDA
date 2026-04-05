#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
import time

from nn import NNetWrapper
from congklak import Congklak
from AlphaZero_mcts import A_MCTS
from parameters import Parameters
from ringbuffer import RingBuffer
import multiprocessing as mp
import torch
import glob
import os
import re
import logging
import json

# Set up basic logging to track training progress without scrolling past 
# thousands of lines of output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Train():
    def __init__(self):
        self.params = Parameters()
        self.comp_time = 0
        self.net = NNetWrapper(params=self.params)
        self.total_training_time = 0
        self.start_iter = 0
        
        # Load persistent training metadata if it exists
        self.meta_path = "training_meta.json"
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r") as f:
                    meta = json.load(f)
                    self.total_training_time = meta.get("total_training_time", 0)
            except Exception as e:
                logging.warning(f"Could not load training metadata: {e}")

        # Automatically load the latest checkpoint
        try:
            # 1. Find highest numbered checkpoint in the folder
            max_num_iter = -1
            current_dir = os.getcwd()
            logging.info(f"Searching for checkpoints in: {current_dir}")
            for f in glob.glob("checkpoint_*.model"):
                match = re.search(r'checkpoint_(\d+).model', f)
                if match:
                    it = int(match.group(1))
                    if it > max_num_iter:
                        max_num_iter = it

            # 2. Check the default 'checkpoint.model'
            default_iter = -1
            try:
                default_iter = self.net.load_checkpoint()
            except (FileNotFoundError, RuntimeError):
                pass

            # 3. Load the newest one
            if max_num_iter > default_iter:
                logging.info(f"Found newer numbered checkpoint. Loading checkpoint_{max_num_iter}.model...")
                self.start_iter = self.net.load_checkpoint(max_num_iter)
            elif default_iter > -1:
                self.start_iter = default_iter
                logging.info(f"Loaded 'checkpoint.model' (iteration {self.start_iter}).")
            else:
                logging.info("No existing checkpoint found. Starting training from scratch.")
                
        except Exception as e:
            print(f"Warning: Error checking checkpoints ({e}). Starting from scratch.")

        # Snapshot parameters to track adjustments
        self.save_hyperparams()

    def save_hyperparams(self):
        """Saves current parameters to a JSON file for experiment tracking."""
        # Filter out non-serializable properties (like modules or classes)
        p_dict = {k: v for k, v in self.params.__dict__.items() 
                  if isinstance(v, (int, float, str, list, bool, dict))}
        
        with open("hyperparameters.json", "w") as f:
            json.dump(p_dict, f, indent=4)
        logging.info("Hyperparameters tracked in 'hyperparameters.json'")

    def Make_schedule(self, num, players):
        num_players = len(players)
        schedule = []
        if players[0] == players[1] and len(players) == 2:
            for n in range(num):
                schedule.append([players[0], players[1]])
        else:
            for n in range(num//2):
                for i in range(num_players):
                    for j in range(num_players):
                        if i != j:
                            schedule.append([players[i], players[j]])
        return schedule

    def AlphaZero(self, g, count):
        amcts = A_MCTS(game=g, net=self.net, params=self.params)
        amcts.num_moves = count
        action = amcts.Run()
        prob = amcts.Get_prob()
        return action, prob

    def Action(self, g, count=0, player="alphazero"):
        if player == "alphazero":
            return self.AlphaZero(g, count)
        else:
            # AlphaZero training should not use Classical MCTS rollouts.
            # Random moves are used if an opponent baseline is needed during training.
            valid_moves = g.Get_valid_moves()
            action = np.random.choice(valid_moves)
            prob = np.zeros(self.params.action_size)
            prob[action] = 1
            return action, prob

    def Run(self):

        buf_board = RingBuffer(self.params.input_size)
        buf_prob = RingBuffer(self.params.input_size)
        buf_v = RingBuffer(self.params.input_size)

        schedule = self.Make_schedule(self.params.num_games, ["alphazero", "alphazero"])
        num_workers = self.params.num_processes_training
        schedule_list = [schedule[i::num_workers] for i in range(num_workers)]

        # Define the absolute target iteration
        target_iter = self.params.num_iterations
        
        print(f"Training started: {self.start_iter} -> {target_iter}")

        for i in range(self.start_iter + 1, target_iter + 1):
            start = time.time()
            training_board = []
            training_prob  = []
            training_v     = np.empty(0)

            devices = self.params.devices
            pool = mp.Pool(num_workers)
            try:
                results = [pool.apply_async(self.self_play, args=(devices[j % len(devices)], schedule_list[j],)) for j in range(num_workers)]
                output = [p.get() for p in results]
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise

            # Reverting to original author's concatenation style
            training_board = []
            training_prob  = []
            for j in range(num_workers):
                training_board += output[j][0]
                training_prob += output[j][1]
                training_v = np.append(training_v, output[j][2])

            num_new_samples = len(training_board)
            logging.info(f"Iteration {i}: Adding {num_new_samples} samples to buffer.")
            
            for j in range(num_new_samples):
                buf_board.add(training_board[j])
                buf_prob.add(training_prob[j])
                buf_v.add(training_v[j])

            self.Learning(buf_board.Get_buffer_start_end(), buf_prob.Get_buffer_start_end(), buf_v.Get_buffer_start_end(), i)
            self.comp_time = time.time() - start
            self.total_training_time += self.comp_time
            avg_time = self.total_training_time / (i - self.start_iter)
            logging.info(f"Iteration {i} completed in {self.comp_time:.2f}s (Avg: {avg_time:.2f}s). Total Elapsed: {self.total_training_time/3600:.2f}h")
            
            # Save persistent metadata
            try:
                with open(self.meta_path, "w") as f:
                    json.dump({"total_training_time": self.total_training_time}, f)
            except Exception as e:
                logging.warning(f"Could not save training metadata: {e}")

            if i % self.params.checkpoint_interval == 0:
                print(f"Checkpoint saved at iteration {i}")

    def self_play(self, device, schedule):
        try:
            self.net.device = device
            self.net.to_device()
            g = Congklak()
            
            board_data = []
            prob_actions = []
            v_data = np.empty(0)
            
            for p in schedule:
                g.Ini_board()
                game_states = []
                game_probs = []
                players_per_step = []
                
                while not g.Check_game_end():
                    game_states.append(g.Get_states())
                    players_per_step.append(g.current_player)
                    action, prob = self.Action(g=g, count=len(game_states), player=p[(len(game_states)-1)%2])
                    game_probs.append(prob)
                    g.Play_action(action)
                
                winner = g.Get_winner()
                for step_player in players_per_step:
                    if winner == 0:
                        v_data = np.append(v_data, 0)
                    elif winner == step_player:
                        v_data = np.append(v_data, 1)
                    else:
                        v_data = np.append(v_data, -1)
                board_data += game_states
                prob_actions += game_probs
                
            return (board_data, prob_actions, v_data)
        except KeyboardInterrupt:
            return ([], [], np.empty(0))

    def Learning(self, training_board, training_prob, training_v, i):
        self.net.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net.to_device()
        self.net.train(np.array(training_board), np.array(training_prob), np.array(training_v))
        
        # Save 'checkpoint.model' every iteration for safety, 
        # butNumbered ones only at specific intervals to manage disk space.
        save_idx = i if i % self.params.checkpoint_interval == 0 else -1
        self.net.save_checkpoint(save_idx)

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    tr = Train()
    try:
        tr.Run()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting cleanly.")
