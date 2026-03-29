using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Threading.Tasks;

namespace CongklakAI
{
    public class CongklakGameController : MonoBehaviour
    {
        [Header("AI Configuration")]
        public AIBrain aiBrain;
        public bool isP1Human = true;
        public bool isP2Human = false;
        
        [Header("Difficulty Parameters")]
        public float sensitivityA = 1.0f;
        public float offsetX0 = 0.0f;
        public int maxSims = 300;

        private CongklakEngine game;
        private int turnCount = 0;
        private bool isInteracting = false;

        void Start()
        {
            game = new CongklakEngine(7);
            StartCoroutine(GameLoop());
        }

        IEnumerator GameLoop()
        {
            while (!game.CheckGameEnd())
            {
                turnCount++;
                bool isHuman = (game.currentPlayer == 1 && isP1Human) || (game.currentPlayer == -1 && isP2Human);

                if (isHuman)
                {
                    Debug.Log($"[Game] Turn {turnCount}: Waiting for Human (P{(game.currentPlayer == 1 ? "1" : "2")})...");
                    isInteracting = true;
                    yield return new WaitUntil(() => !isInteracting);
                }
                else
                {
                    Debug.Log($"[Game] Turn {turnCount}: AI (P{(game.currentPlayer == 1 ? "1" : "2")}) is thinking...");
                    
                    // Run MCTS in a separate thread to avoid frame freezing
                    int aiMove = -1;
                    Task.Run(() => {
                        AlphaDDA_MCTS mcts = new AlphaDDA_MCTS(game, aiBrain, sensitivityA, offsetX0, maxSims);
                        aiMove = mcts.Run(turnCount);
                    }).Wait(); // For simplicity in this demo, though async/await is better

                    yield return new WaitForSeconds(0.5f); // Cosmetic delay
                    ExecuteMove(aiMove);
                }

                yield return null;
            }

            Debug.Log($"[Game] Game Over! Winner: P{(game.winner == 1 ? "1" : "2")}");
        }

        /// <summary>
        /// This method should be called by your UI Buttons/Holes when clicked.
        /// </summary>
        /// <param name="relativeHoleIdx">0-6 for current player's side</param>
        public void OnHoleClicked(int relativeHoleIdx)
        {
            if (!isInteracting) return;

            List<int> validMoves = game.GetValidMoves();
            if (validMoves.Contains(relativeHoleIdx))
            {
                isInteracting = false;
                ExecuteMove(relativeHoleIdx);
            }
            else
            {
                Debug.LogWarning("Invalid move!");
            }
        }

        private void ExecuteMove(int move)
        {
            Debug.Log($"[Game] Executing move {move} for P{(game.currentPlayer == 1 ? "1" : "2")}");
            game.PlayAction(move);
            UpdateUI();
        }

        private void UpdateUI()
        {
            // Placeholder for your UI Refresh logic
            // You would loop through your hole objects and update their shell counts
            /*
            for(int i=0; i<16; i++) {
                holeUI[i].text = game.board[i].ToString();
            }
            */
        }
    }
}
