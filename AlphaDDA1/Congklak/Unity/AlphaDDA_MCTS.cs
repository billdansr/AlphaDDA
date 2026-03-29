using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace CongklakAI
{
    /// <summary>
    /// A single node in the MCTS tree.
    /// </summary>
    public class MCTSNode
    {
        public int nsa = 0;      // Number of visits
        public float wsa = 0;    // Total value
        public float qsa = 0;    // Average value (W/N)
        public float psa = 0;    // Prior probability (from NN)
        
        public int player;       // Which player IS moving in this state
        public int? move;        // The move that led to this state
        public int[] board;      // Board representation
        
        public MCTSNode parent;
        public List<MCTSNode> children = new List<MCTSNode>();
        
        public bool terminal;
        public int winner;

        public MCTSNode(int[] board, int player, int? move = null, float psa = 0, bool terminal = false, int winner = 0, MCTSNode parent = null)
        {
            this.board = (int[])board.Clone();
            this.player = player;
            this.move = move;
            this.psa = psa;
            this.terminal = terminal;
            this.winner = winner;
            this.parent = parent;
        }

        public void AddChild(int[] nextBoard, int nextPlayer, int move, float psa, bool terminal, int winner)
        {
            children.Add(new MCTSNode(nextBoard, nextPlayer, move, psa, terminal, winner, this));
        }
    }

    /// <summary>
    /// AlphaDDA Monte Carlo Tree Search implementation.
    /// </summary>
    public class AlphaDDA_MCTS
    {
        private MCTSNode root;
        private AIBrain brain;
        
        // Hyperparameters (matched to parameters.py)
        private float cpuct = 1.25f;
        private float rnd_rate = 0.2f;
        private float temp = 50f;
        private int openingLimit = 0; // opening_test = 0

        // DDA parameters
        private int N_MAX = 300;
        private float A = 1.0f;
        private float X0 = 0.0f;

        public AlphaDDA_MCTS(CongklakEngine game, AIBrain brain, float A = 1.0f, float X0 = 0.0f, int N_MAX = 300)
        {
            this.root = new MCTSNode(game.board, game.currentPlayer);
            this.brain = brain;
            this.A = A;
            this.X0 = X0;
            this.N_MAX = N_MAX;
        }

        /// <summary>
        /// Runs the MCTS and returns the best move.
        /// </summary>
        public int Run(int turnCount)
        {
            // 1. Calculate Dynamic Simulations
            var (pi_root, v_root) = brain.Predict(GenerateStates(root.board, root.player));
            
            float winScore = v_root; // Current state evaluation
            float exponent = -A * (winScore + X0) + (float)Math.Log10(N_MAX / 2.0);
            int numSims = (int)Math.Ceiling(Math.Pow(10, exponent));
            numSims = Math.Clamp(numSims, 1, N_MAX);
            
            Debug.Log($"[AlphaDDA] Sims: {numSims}, AI Confidence: {winScore:F3}");

            // 2. MCTS Logic
            for (int i = 0; i < numSims; i++)
            {
                MCTSNode node = root;
                
                // Selection phase: crawl down the tree using PUCT
                while (node.children.Count > 0 && !node.terminal)
                {
                    node = SelectChild(node);
                }

                float v;
                if (node.terminal)
                {
                    v = node.winner;
                }
                else
                {
                    // Expansion phase: ask NN for priors and value
                    var (pi, val) = brain.Predict(GenerateStates(node.board, node.player));
                    v = val;
                    ExpandNode(node, pi);
                }

                // Backpropagation
                Backpropagate(node, v);
            }

            return DecideMove(turnCount);
        }

        private MCTSNode SelectChild(MCTSNode node)
        {
            int totalN = node.children.Sum(c => c.nsa);
            
            MCTSNode best = null;
            float bestValue = float.NegativeInfinity;

            // Handle random exploration at root (rnd_rate)
            if (node == root && UnityEngine.Random.value < rnd_rate)
            {
                return node.children[UnityEngine.Random.Range(0, node.children.Count)];
            }

            foreach (var child in node.children)
            {
                // PUCT Formula: Q + cpuct * P * sqrt(N) / (1 + n)
                float u = child.qsa + cpuct * child.psa * (float)Math.Sqrt(totalN) / (1 + child.nsa);
                if (u > bestValue)
                {
                    bestValue = u;
                    best = child;
                }
            }
            return best;
        }

        private void ExpandNode(MCTSNode node, float[] piVector)
        {
            CongklakEngine simEngine = new CongklakEngine();
            Array.Copy(node.board, simEngine.board, 16);
            simEngine.currentPlayer = node.player;

            List<int> validMoves = simEngine.GetValidMoves();
            
            // Mask and normalize policy
            float sumPsa = 0;
            foreach (int move in validMoves) sumPsa += piVector[move];

            foreach (int move in validMoves)
            {
                float psa = (sumPsa > 0) ? piVector[move] / sumPsa : 1.0f / validMoves.Count;
                
                // Simulate move to create child state
                simEngine.InitializeBoard(7); // Reset temp to reuse logic
                Array.Copy(node.board, simEngine.board, 16);
                simEngine.currentPlayer = node.player;
                
                simEngine.PlayAction(move);
                
                node.AddChild(
                    simEngine.board, 
                    simEngine.currentPlayer, 
                    move, 
                    psa, 
                    simEngine.CheckGameEnd(), 
                    simEngine.winner
                );
            }
        }

        private void Backpropagate(MCTSNode node, float v)
        {
            MCTSNode curr = node;
            while (curr != null)
            {
                curr.nsa++;
                curr.wsa += v;
                curr.qsa = curr.wsa / curr.nsa;
                
                // Flip value perspective if parent player is different
                if (curr.parent != null && curr.parent.player != curr.player)
                {
                    v = -v;
                }
                curr = curr.parent;
            }
        }

        private int DecideMove(int turnCount)
        {
            if (turnCount > openingLimit)
            {
                // Deterministic: Max visits
                return root.children.OrderByDescending(c => c.nsa).First().move.Value;
            }
            else
            {
                // Softmax exploration for opening moves
                float[] visits = root.children.Select(c => (float)c.nsa).ToArray();
                float[] probs = Softmax(visits);
                
                double r = UnityEngine.Random.value;
                double cumulative = 0;
                for (int i = 0; i < probs.Length; i++)
                {
                    cumulative += probs[i];
                    if (r <= cumulative) return root.children[i].move.Value;
                }
                return root.children.First().move.Value;
            }
        }

        private float[] Softmax(float[] x)
        {
            float[] exp = x.Select(v => (float)Math.Exp(v / temp)).ToArray();
            float sum = exp.Sum();
            return exp.Select(v => v / sum).ToArray();
        }

        // Static helper to generate canonical state for the AI
        public static float[,,] GenerateStates(int[] board, int currentPlayer)
        {
            float[,,] states = new float[3, 2, 8];
            
            // Canonical indices for Row 0 (Current Player) and Row 1 (Opponent)
            int[] mySide = (currentPlayer == 1) ? Enumerable.Range(0, 8).ToArray() : Enumerable.Range(8, 8).ToArray();
            int[] opSide = (currentPlayer == 1) ? Enumerable.Range(8, 8).ToArray() : Enumerable.Range(0, 8).ToArray();

            for (int j = 0; j < 8; j++)
                states[0, 0, j] = board[mySide[j]] / 98.0f;

            for (int j = 0; j < 8; j++)
                states[1, 1, j] = board[opSide[j]] / 98.0f;

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 8; j++)
                    states[2, i, j] = 1.0f;

            return states;
        }
    }
}
