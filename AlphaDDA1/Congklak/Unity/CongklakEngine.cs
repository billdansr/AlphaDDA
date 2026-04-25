using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace CongklakAI
{
    public class CongklakEngine
    {
        public const int BOARD_SIZE = 16;
        public const int ACTION_SIZE = 7;
        
        // 0-6: P1 Houses, 7: P1 Store, 8-14: P2 Houses, 15: P2 Store
        public int[] board = new int[BOARD_SIZE];
        public int currentPlayer = 1; // 1 for P1, -1 for P2
        public int winner = 0;
        
        public CongklakEngine(int initialShells = 7)
        {
            InitializeBoard(initialShells);
        }

        public void InitializeBoard(int initialShells)
        {
            Array.Clear(board, 0, BOARD_SIZE);
            for (int i = 0; i < 7; i++)
            {
                board[i] = initialShells;
                board[i + 8] = initialShells;
            }
            currentPlayer = 1;
            winner = 0;
        }

        public List<int> GetValidMoves()
        {
            List<int> moves = new List<int>();
            if (currentPlayer == 1)
            {
                for (int i = 0; i < 7; i++)
                    if (board[i] > 0) moves.Add(i);
            }
            else
            {
                for (int i = 0; i < 7; i++)
                    if (board[i + 8] > 0) moves.Add(i);
            }
            return moves;
        }

        public bool CheckGameEnd()
        {
            bool p1Empty = true;
            for (int i = 0; i < 7; i++) if (board[i] > 0) { p1Empty = false; break; }
            
            bool p2Empty = true;
            for (int i = 8; i < 15; i++) if (board[i] > 0) { p2Empty = false; break; }

            if (p1Empty || p2Empty)
            {
                // Sweep remaining shells to stores
                if (p1Empty)
                {
                    for (int i = 8; i < 15; i++) { board[15] += board[i]; board[i] = 0; }
                }
                if (p2Empty)
                {
                    for (int i = 0; i < 7; i++) { board[7] += board[i]; board[i] = 0; }
                }

                if (board[7] > board[15]) winner = 1;
                else if (board[15] > board[7]) winner = -1;
                else winner = 0; // Draw
                
                return true;
            }
            return false;
        }

        public IEnumerable<(int holeIdx, int shellsInHand)> PlayAction(int action)
        {
            // Action is relative (0..6)
            int startHole = (currentPlayer == 1) ? action : action + 8;
            
            int shells = board[startHole];
            board[startHole] = 0;
            int currHole = startHole;
            
            bool extraTurn = false;
            
            while (shells > 0)
            {
                currHole = (currHole + 1) % 16;
                
                // Skip opponent's store
                if (currentPlayer == 1 && currHole == 15) continue;
                if (currentPlayer == -1 && currHole == 7) continue;
                
                // Drop 1 shell
                board[currHole]++;
                shells--;
                
                // YIELD for UI animation: (where we dropped, how many left in hand)
                yield return (currHole, shells);
                
                if (shells == 0)
                {
                    // 1. Drops in own store -> extra turn
                    if ((currentPlayer == 1 && currHole == 7) || (currentPlayer == -1 && currHole == 15))
                    {
                        extraTurn = true;
                        break;
                    }
                    // 2. Drops in empty hole on own side -> capture
                    else if (board[currHole] == 1)
                    {
                        if (IsOwnHole(currHole, currentPlayer))
                        {
                            int opposite = 14 - currHole;
                            if (board[opposite] > 0)
                            {
                                int captured = board[opposite] + 1;
                                board[opposite] = 0;
                                board[currHole] = 0;
                                
                                int storeIdx = (currentPlayer == 1) ? 7 : 15;
                                board[storeIdx] += captured;
                                
                                // YIELD a special signal for capture animation
                                yield return (currHole, 0); 
                            }
                        }
                        break; // End of turn
                    }
                    // 3. Drops in non-empty hole -> pick up and continue
                    else if (board[currHole] > 1)
                    {
                        shells = board[currHole];
                        board[currHole] = 0;
                        
                        // YIELD for UI animation: Picking up shells
                        yield return (currHole, shells);
                    }
                }
            }
            
            if (!extraTurn)
                currentPlayer *= -1;
        }

        private bool IsOwnHole(int idx, int player)
        {
            if (player == 1) return idx >= 0 && idx <= 6;
            return idx >= 8 && idx <= 14;
        }

        public float[,,] GetStates()
        {
            // Redirect to synchronized central logic in AlphaDDA_MCTS
            return AlphaDDA_MCTS.GenerateStates(this.board, this.currentPlayer);
        }

        private void StatesSet(float[,,] s, int channel, int x, int y, float v)
        {
            s[channel, x, y] = v;
        }

        public CongklakEngine Clone()
        {
            CongklakEngine clone = new CongklakEngine();
            Array.Copy(this.board, clone.board, BOARD_SIZE);
            clone.currentPlayer = this.currentPlayer;
            clone.winner = this.winner;
            return clone;
        }
    }
}
