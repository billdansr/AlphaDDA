using System;
using System.Collections;
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
        private float temp = 1.0f;
        private int openingLimit = 0; // opening_test = 0

        // DDA parameters
        private int N_MAX = 800;
        private int N_MIN = 60;
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
        /// Runs the MCTS as a coroutine to keep inference on the main thread while preventing UI freezes.
        /// </summary>
        public IEnumerator RunCoroutine(int turnCount, Action<int> onComplete)
        {
            if (onComplete == null) yield break;

            // 1. Calculate Dynamic Simulations
            // 1. DDA: Adjust simulations based on AI confidence
            // Sync with Python AlphaDDA1.py: squared reduction formula
            var (pi_root, v_root) = brain.Predict(GenerateStates(root.board, root.player));
            float winScore = v_root;
            int numSims;
            if (winScore > 0)
            {
                // Reduce simulations when AI is winning (smoother squared reduction)
                float reductionFactor = 1.0f - (winScore * winScore);
                numSims = Mathf.Max(N_MIN, (int)(N_MAX * reductionFactor));
            }
            else
            {
                numSims = N_MAX;
            }
            Debug.Log($"[AlphaDDA] v={winScore:F3} -> Sims: {numSims}");

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
                    // Value must be relative to the player at the node
                    if (node.winner == 0) v = 0;
                    else v = (node.winner == node.player) ? 1.0f : -1.0f;
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

                // Time-slicing: yield every 20 simulations to keep the UI responsive
                if (i % 20 == 0) yield return null;
            }

            onComplete(DecideMove(turnCount));
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
            // Create a temporary engine to simulate moves
            CongklakEngine simEngine = new CongklakEngine(0);
            Array.Copy(node.board, simEngine.board, 16);
            simEngine.currentPlayer = node.player;

            List<int> validMoves = simEngine.GetValidMoves();
            
            // Jika pemain tidak punya langkah tapi game belum berakhir, ganti pemain (Pass)
            if (validMoves.Count == 0 && !simEngine.CheckGameEnd())
            {
                simEngine.currentPlayer *= -1;
                validMoves = simEngine.GetValidMoves();
            }

            if (validMoves.Count == 0) return; 
            
            // Mask and normalize policy
            float sumPsa = 0;
            foreach (int move in validMoves) sumPsa += piVector[move];

            foreach (int move in validMoves)
            {
                float psa = (sumPsa > 0) ? piVector[move] / sumPsa : 1.0f / validMoves.Count;
                
                // Simulate move to create child state
                Array.Copy(node.board, simEngine.board, 16);
                simEngine.currentPlayer = node.player;
                
                foreach (var _ in simEngine.PlayAction(move)) { } // Execute instantly
                
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
            // Stop before root, matching Python: 'while node != self.root'
            while (curr != null && curr != root)
            {
                // Flip perspective if parent player changed (handles extra turns correctly)
                if (curr.parent != null && curr.parent.player != curr.player)
                {
                    v = -v;
                }

                curr.nsa++;
                curr.wsa += v;
                curr.qsa = curr.wsa / curr.nsa;
                
                curr = curr.parent;
            }
        }

        private int DecideMove(int turnCount)
        {
            if (root.children.Count == 0) return -1;

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
            
            // Sync with Python: P1 = 1.0, P2 = 0.0
            float turnIndicator = (currentPlayer == 1) ? 1.0f : 0.0f;

            int myOffset = (currentPlayer == 1) ? 0 : 8;
            int opOffset = (currentPlayer == 1) ? 8 : 0;

            for (int j = 0; j < 8; j++) 
            {
                // Channel 0: Current player's shells on their canonical row (Row 0)
                states[0, 0, j] = board[myOffset + j] / 98.0f;
                
                // Channel 1: Opponent's shells on their canonical row (Row 1)
                states[1, 1, j] = board[opOffset + j] / 98.0f;
                
                // Channel 2: Perspective/Turn metadata
                states[2, 0, j] = states[2, 1, j] = turnIndicator;
            }

            return states;
        }
    }
}
