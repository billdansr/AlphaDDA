using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

namespace CongklakAI
{
    public class CongklakUnityTester : MonoBehaviour
    {
        void Start()
        {
            RunTests();
        }

        void RunTests()
        {
            Debug.Log("--- Starting Congklak Logic Tests ---");
            
            TestSowing();
            TestCapture();
            TestExtraTurn();
            TestCanonicalization();
            
            Debug.Log("--- All Tests Completed ---");
        }

        void TestSowing()
        {
            CongklakEngine engine = new CongklakEngine(1); // 1 shell per hole
            foreach (var _ in engine.PlayAction(0)) { } // Execute instantly
            
            // Should have 0 in hole 0, and 1 in hole 1 (from sowing) 
            // Wait, if it had 1 shell, it drops it in hole 1. Then hole 1 had 1, now has 2.
            // In Congklak rules, if you drop in a non-empty hole, you continue sowing (Jalan Terus).
            // This test verifies 'Jalan Terus' logic.
            
            Debug.Log($"[TestSowing] P1 Store after move 0: {engine.board[7]} (Expected > 0 if it kept sowing)");
        }

        void TestCapture()
        {
            CongklakEngine engine = new CongklakEngine(0);
            engine.board[0] = 1;      // One shell to move
            engine.board[1] = 0;      // Landing spot (empty)
            engine.board[13] = 10;    // Opposite hole to capture (Opposite of 1 is 14-1=13)
            
            foreach (var _ in engine.PlayAction(0)) { }
            
            if (engine.board[7] == 11) // 10 from opposite + 1 from landing
                Debug.Log("✅ TestCapture: Passed");
            else
                Debug.LogError($"❌ TestCapture: Failed (Store: {engine.board[7]}, Expected 11)");
        }

        void TestExtraTurn()
        {
            CongklakEngine engine = new CongklakEngine(0);
            engine.board[6] = 1; // Last hole before P1 Store (7)
            
            foreach (var _ in engine.PlayAction(6)) { }
            
            if (engine.currentPlayer == 1)
                Debug.Log("✅ TestExtraTurn: Passed (P1 kept the turn)");
            else
                Debug.LogError("❌ TestExtraTurn: Failed (Turn switched to P2)");
        }

         void TestCanonicalization()
        {
            // Scenario: P1 has 5 shells in their first hole (idx 0).
            int[] boardP1 = new int[16];
            boardP1[0] = 5;
            float[,,] stateP1 = AlphaDDA_MCTS.GenerateStates(boardP1, 1);

            // Scenario: P2 has 5 shells in their first hole (idx 8).
            int[] boardP2 = new int[16];
            boardP2[8] = 5;
            float[,,] stateP2 = AlphaDDA_MCTS.GenerateStates(boardP2, -1);

            // Both states should be identical because the AI should always see "self" at Row 0.
            bool match = true;
            for (int c = 0; c < 3; c++)
                for (int x = 0; x < 2; x++)
                    for (int y = 0; y < 8; y++)
                        if (Mathf.Abs(stateP1[c, x, y] - stateP2[c, x, y]) > 0.0001f)
                            match = false;

            if (match)
                Debug.Log("✅ TestCanonicalization: Passed (Perspective is identical for both players)");
            else
            {
                Debug.LogError("❌ TestCanonicalization: Failed! The AI sees different inputs for identical relative board positions.");
            }
        }
    }
}
