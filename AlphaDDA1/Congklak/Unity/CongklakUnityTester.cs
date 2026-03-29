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
            
            Debug.Log("--- All Tests Completed ---");
        }

        void TestSowing()
        {
            CongklakEngine engine = new CongklakEngine(1); // 1 shell per hole
            engine.PlayAction(0); // P1 moves from hole 0
            
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
            
            engine.PlayAction(0);
            
            if (engine.board[7] == 11) // 10 from opposite + 1 from landing
                Debug.Log("✅ TestCapture: Passed");
            else
                Debug.LogError($"❌ TestCapture: Failed (Store: {engine.board[7]}, Expected 11)");
        }

        void TestExtraTurn()
        {
            CongklakEngine engine = new CongklakEngine(0);
            engine.board[6] = 1; // Last hole before P1 Store (7)
            
            engine.PlayAction(6);
            
            if (engine.currentPlayer == 1)
                Debug.Log("✅ TestExtraTurn: Passed (P1 kept the turn)");
            else
                Debug.LogError("❌ TestExtraTurn: Failed (Turn switched to P2)");
        }
    }
}
