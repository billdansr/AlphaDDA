using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Threading.Tasks;
using UnityEngine.InputSystem;
using TMPro;

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
        public float stepDelay = 0.2f; // Time between shell drops

        [Header("Visual Tuning")]
        public float holeRadius = 0.15f; // Radius penyebaran biji di dalam lubang
        public bool useRandomRotation = true;

        [Header("Animation Settings")]
        public GameObject shellPrefab;   // Assign a small shell/circle prefab
        public float shellMoveSpeed = 15f; // Speed of the shell moving between holes
        public Transform p1HandTarget;    // Titik di dekat area bawah (P1)
        public Transform p2HandTarget;    // Titik di dekat area atas (P2)

        [Header("Audio Settings")]
        public AudioSource audioSource;
        public AudioSource musicSource;
        public AudioClip swooshSound;
        public AudioClip dropSound;
        public AudioClip bgMusic;

        [Header("UI References")]
        public TMP_Text handShellCounterText; // New: Text to display shell count in hand
        public TMP_Text[] holeTexts; // Using TextMeshPro for higher quality
        public TMP_Text statusText;  // Optional: To show whose turn it is
        public Transform[] holeTransforms; // Drag your 16 Sprite Holes here

        private CongklakEngine game;
        private int turnCount = 0;
        private bool isInteracting = false;
        private int pendingMove = -1;
        private Camera mainCamera;
        private List<GameObject>[] holeShells;
        private Vector3 shellBaseScale = Vector3.one;

        void Start()
        {
            // Inisialisasi list untuk menampung objek shell di setiap lubang
            holeShells = new List<GameObject>[16];

            // Cek apakah array sudah di-assign di Inspector
            if (holeTransforms == null || holeTransforms.Length != 16 || holeTexts == null || holeTexts.Length != 16)
            {
                Debug.LogError("[Game] Setup Inspector belum lengkap! Pastikan holeTexts dan holeTransforms berjumlah 16.");
                this.enabled = false;
                return;
            }

            for (int i = 0; i < 16; i++) holeShells[i] = new List<GameObject>();

            if (shellPrefab != null)
                shellBaseScale = shellPrefab.transform.localScale;

            mainCamera = Camera.main;
            game = new CongklakEngine(7);
            UpdateAllHoleVisuals(); // Spawn shell sesuai initial shells
            UpdateUI(); 
            AlignUIToWorld(); // Snap UI to the sprites
            
            // Inisialisasi dan putar musik latar
            if (musicSource != null && bgMusic != null && musicSource.clip != bgMusic)
            {
                musicSource.clip = bgMusic;
                musicSource.loop = true;
                musicSource.Play();
            }
            else if (musicSource != null && !musicSource.isPlaying)
            {
                musicSource.Play();
            }

            if (handShellCounterText != null)
            {
                handShellCounterText.gameObject.SetActive(false); // Hide initially
            }

            StartCoroutine(GameLoop());
        }

        void Update()
        {
            // Detect clicks on World Space Sprites using the New Input System
            if (isInteracting && Pointer.current != null && Pointer.current.press.wasPressedThisFrame)
            {
                Vector2 mousePosition = Pointer.current.position.ReadValue();
                Vector2 worldPosition = mainCamera.ScreenToWorldPoint(mousePosition);
                Collider2D hit = Physics2D.OverlapPoint(worldPosition);

                if (hit != null)
                {
                    // Optimization: Check for a specific naming convention or tag
                    string objName = hit.gameObject.name;
                    
                    // Handle names starting with "Hole_" or "Store_"
                    int absoluteIdx = -1;
                    bool isHole = objName.StartsWith("Hole_") && int.TryParse(objName.Substring(5), out absoluteIdx);
                    
                    if (isHole)
                    {
                        // Map absolute board index back to relative index (0-6)
                        int relativeIdx = -1;
                        if (game.currentPlayer == 1 && absoluteIdx >= 0 && absoluteIdx <= 6)
                            relativeIdx = absoluteIdx;
                        else if (game.currentPlayer == -1 && absoluteIdx >= 8 && absoluteIdx <= 14)
                            relativeIdx = absoluteIdx - 8;

                        if (relativeIdx != -1) OnHoleClicked(relativeIdx);
                    }
                }
            }
        }

        IEnumerator GameLoop()
        {
            while (!game.CheckGameEnd())
            {
                turnCount++;
                
                // Cek apakah pemain saat ini punya langkah. Jika tidak, lompati giliran (Pass)
                if (game.GetValidMoves().Count == 0)
                {
                    string skipper = game.currentPlayer == 1 ? "P1" : "P2";
                    SetStatus($"{skipper} kosong, melompati giliran...");
                    game.currentPlayer *= -1;
                    yield return new WaitForSeconds(1.0f);
                    continue;
                }

                bool isHuman = (game.currentPlayer == 1 && isP1Human) || (game.currentPlayer == -1 && isP2Human);

                if (isHuman)
                {
                    SetStatus($"P{(game.currentPlayer == 1 ? "1" : "2")} (Human) Turn");
                    pendingMove = -1;
                    isInteracting = true;
                    yield return new WaitUntil(() => pendingMove != -1);
                    yield return StartCoroutine(ExecuteMove(pendingMove));
                }
                else
                {
                    SetStatus($"AI (P{(game.currentPlayer == 1 ? "1" : "2")}) Thinking...");

                    int aiMove = -1;
                    AlphaDDA_MCTS mcts = new AlphaDDA_MCTS(game, aiBrain, sensitivityA, offsetX0, maxSims);
                    
                    // Run MCTS on main thread via Coroutine (Inference safe)
                    yield return StartCoroutine(mcts.RunCoroutine(turnCount, (move) => aiMove = move));

                    yield return new WaitForSeconds(0.5f); // Cosmetic delay
                    if (aiMove != -1)
                        yield return StartCoroutine(ExecuteMove(aiMove));
                    else
                        Debug.LogError("AI failed to return a valid move!");
                }

                yield return null;
            }

            SetStatus($"Game Over! Winner: P{(game.winner == 1 ? "1" : "2")}");
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
                pendingMove = relativeHoleIdx;
                isInteracting = false;
            }
            else
            {
                Debug.LogWarning("Invalid move!");
            }
        }

        private IEnumerator ExecuteMove(int move)
        {
            if (shellPrefab == null)
            {
                Debug.LogError("[Game] Shell Prefab belum di-assign di Inspector!");
                yield break;
            }

            GameObject handShell = null;
            // Z-Offset agar shell muncul di depan papan (misal: Z papan adalah 0, maka Z shell -1)
            float zOffset = -0.5f; 
            int lastHoleIdx = -1;
            int storeIdx = (game.currentPlayer == 1) ? 7 : 15;
            int lastStoreCount = game.board[storeIdx];

            foreach (var (holeIdx, remainingShells) in game.PlayAction(move))
            {
                if (holeTransforms == null || holeIdx >= holeTransforms.Length || holeTransforms[holeIdx] == null)
                {
                    Debug.LogError($"[Game] holeTransforms[{holeIdx}] tidak ditemukan!");
                    continue;
                }

                // Deteksi Aksi Spesial: Jalan Terus (Pick up) atau Capture (Tembak)
                if (holeIdx == lastHoleIdx && game.board[holeIdx] == 0)
                {
                    // Jika jumlah biji di lumbung (store) bertambah, berarti engine melakukan Capture
                    // Karena biji yang jatuh di lumbung secara normal tidak memicu yield ganda.
                    if (game.board[storeIdx] > lastStoreCount)
                    {
                        yield return StartCoroutine(AnimateCapture(holeIdx, handShell));
                        handShell = null; // Shell di tangan sudah masuk ke store
                        if (handShellCounterText != null)
                        {
                            handShellCounterText.gameObject.SetActive(false); // Hide counter after capture
                        }
                    }
                    else
                    {
                        // Jalan Terus: Ambil semua biji di lubang ini ke arah tangan
                        if (handShellCounterText != null)
                        {
                            handShellCounterText.text = remainingShells.ToString();
                        }
                        
                        yield return StartCoroutine(AnimatePickup(holeIdx, handShell, false));
                        
                        // Segera update papan agar lubang yang baru diambil terlihat kosong (0)
                        UpdateAllHoleVisuals();
                        UpdateUI();
                    }

                    lastStoreCount = game.board[storeIdx];
                    continue;
                }

                Vector3 targetPos = holeTransforms[holeIdx].position;
                targetPos.z += zOffset; // Berikan offset agar tidak tumpang tindih secara Z

                if (handShell == null)
                {
                    // Inisialisasi tangan sebagai wadah kosong (Container)
                    handShell = new GameObject("HandGroup");
                    handShell.transform.position = targetPos;

                    if (handShellCounterText != null)
                    {
                        handShellCounterText.text = remainingShells.ToString();
                        handShellCounterText.gameObject.SetActive(true);
                    }
                    // Animasi ambil biji dari lubang awal ke arah tangan
                    yield return StartCoroutine(AnimatePickup(holeIdx, handShell, true));

                    // Sinkronisasi visual lubang agar langsung terlihat kosong saat diambil
                    UpdateAllHoleVisuals();
                    UpdateUI();
                }
                else
                {
                    if (handShellCounterText != null)
                    {
                        handShellCounterText.text = remainingShells.ToString();
                    }
                    // Move the shell from the previous hole to the current one
                    Vector3 startPos = handShell.transform.position;
                    float distance = Vector3.Distance(startPos, targetPos);
                    float duration = distance / shellMoveSpeed;
                    float elapsed = 0;

                    while (elapsed < duration)
                    {
                        handShell.transform.position = Vector3.Lerp(startPos, targetPos, elapsed / duration);
                        elapsed += Time.deltaTime;
                        yield return null;
                    }
                    handShell.transform.position = targetPos;

                    // Visual drop: Hapus biji dari tangan agar sesuai dengan jumlah di engine
                    if (handShell.transform.childCount > remainingShells)
                    {
                        int toDestroy = handShell.transform.childCount - remainingShells;
                        for (int i = 0; i < toDestroy; i++) 
                            Destroy(handShell.transform.GetChild(0).gameObject);
                    }
                    
                    // Mainkan suara klik saat biji sampai di lubang
                    if (audioSource != null && dropSound != null)
                        audioSource.PlayOneShot(dropSound);

                    // Update visual setelah shell "jatuh" ke lubang
                    UpdateAllHoleVisuals();
                }

                UpdateUI();
                yield return new WaitForSeconds(stepDelay);
                lastHoleIdx = holeIdx;
                lastStoreCount = game.board[storeIdx];
            }

            if (handShell != null) Destroy(handShell);
            if (handShellCounterText != null)
            {
                handShellCounterText.gameObject.SetActive(false); // Hide counter at the end of the move
            }
        }

        private IEnumerator AnimateCapture(int landingHoleIdx, GameObject lastDroppedShell)
        {
            string playerName = game.currentPlayer == 1 ? "P1" : "P2";
            SetStatus($"{playerName} CAPTURE! (Menembak lawan)");
            
            // Tambahkan jeda awal agar pemain bisa melihat posisi jatuh terakhir
            yield return new WaitForSeconds(0.8f);

            int oppositeIdx = 14 - landingHoleIdx;
            int storeIdx = (game.currentPlayer == 1) ? 7 : 15;
            Transform handTarget = (game.currentPlayer == 1) ? p1HandTarget : p2HandTarget;

            // 1. Ambil semua shell yang ada di lubang lawan (opposite)
            List<GameObject> capturedObjects = new List<GameObject>(holeShells[oppositeIdx]);
            holeShells[oppositeIdx].Clear();
            
            // Tambahkan shell yang baru saja jatuh (yang memicu capture)
            if (lastDroppedShell != null)
            {
                foreach (Transform child in lastDroppedShell.transform)
                    capturedObjects.Add(child.gameObject);
                
                foreach (var obj in capturedObjects) obj.transform.SetParent(null);
            }

            if (capturedObjects.Count == 0) yield break;

            // Play swoosh sound when moving captured shells to hand
            if (audioSource != null && swooshSound != null)
                audioSource.PlayOneShot(swooshSound);

            // 2. Animasi ke Tangan (Hand Target)
            float duration = 0.6f;
            float elapsed = 0;
            Vector3 startHandPos = handTarget != null ? handTarget.position : holeTransforms[storeIdx].position;

            while (elapsed < duration)
            {
                float t = elapsed / duration;
                // Efek Scale Up saat "diambil"
                float scale = Mathf.Lerp(1f, 1.5f, Mathf.Sin(t * Mathf.PI));

                foreach (var obj in capturedObjects)
                {
                    if (obj == null) continue;
                    obj.transform.position = Vector3.Lerp(obj.transform.position, startHandPos, t);
                    obj.transform.localScale = shellBaseScale * scale;
                }
                elapsed += Time.deltaTime;
                yield return null;
            }

            // 3. Animasi dari Tangan ke Store (Lumbung)
            elapsed = 0;
            Vector3 storePos = holeTransforms[storeIdx].position;
            while (elapsed < duration)
            {
                float t = elapsed / duration;
                foreach (var obj in capturedObjects)
                {
                    if (obj == null) continue;
                    obj.transform.position = Vector3.Lerp(startHandPos, storePos, t);
                    obj.transform.localScale = Vector3.Lerp(shellBaseScale * 1.5f, shellBaseScale, t);
                }
                elapsed += Time.deltaTime;
                yield return null;
            }

            // Hapus objek animasi dan sinkronkan visual lumbung yang baru
            foreach (var obj in capturedObjects) Destroy(obj);
            
            // Mainkan suara klik saat kumpulan biji masuk ke lumbung
            if (audioSource != null && dropSound != null)
                audioSource.PlayOneShot(dropSound);

            UpdateAllHoleVisuals();
            UpdateUI();
        }

        private IEnumerator AnimatePickup(int holeIdx, GameObject currentHandShell, bool isInitial)
        {
            if (!isInitial)
            {
                string playerName = game.currentPlayer == 1 ? "P1" : "P2";
                SetStatus($"{playerName} JALAN TERUS!");
                
                // Jeda singkat agar transisi pengambilan biji tidak terlalu mendadak
                yield return new WaitForSeconds(0.4f);
            }

            Transform handTarget = (game.currentPlayer == 1) ? p1HandTarget : p2HandTarget;
            Vector3 targetHandPos = handTarget != null ? handTarget.position : holeTransforms[holeIdx].position + Vector3.up * 1.5f;
            targetHandPos.z = -0.5f;

            // 1. Kumpulkan semua visual shell di lubang tersebut (termasuk yang baru jatuh)
            List<GameObject> pickedObjects = new List<GameObject>(holeShells[holeIdx]);
            holeShells[holeIdx].Clear(); // Kosongkan list agar tidak terhapus ganda oleh sistem visual
            
            if (pickedObjects.Count == 0) yield break;

            // Play swoosh sound when picking up shells
            if (audioSource != null && swooshSound != null)
                audioSource.PlayOneShot(swooshSound);

            // 2. Animasi bergerak ke arah posisi tangan
            float duration = 0.5f;
            float elapsed = 0;
            while (elapsed < duration)
            {
                float t = elapsed / duration;
                float scale = Mathf.Lerp(1f, 1.4f, t);

                foreach (var obj in pickedObjects)
                {
                    if (obj == null) continue;
                    obj.transform.position = Vector3.Lerp(obj.transform.position, targetHandPos, t);
                    obj.transform.localScale = shellBaseScale * scale;
                }
                elapsed += Time.deltaTime;
                yield return null;
            }

            // 3. Pindahkan objek ke dalam wadah tangan (Container)
            currentHandShell.transform.position = targetHandPos;
            foreach (var obj in pickedObjects)
            {
                obj.transform.SetParent(currentHandShell.transform);
                
                // Berikan posisi acak lokal agar terlihat seperti tumpukan di tangan
                obj.transform.localPosition = (Vector3)(Random.insideUnitCircle * 0.2f);
                obj.transform.localScale = shellBaseScale;

                // Pastikan biji di tangan berada paling depan secara visual
                if (obj.TryGetComponent<SpriteRenderer>(out var sr)) sr.sortingOrder = 20;
            }

            yield return new WaitForSeconds(0.1f);
        }

        /// <summary>
        /// Snaps the UI text elements to the screen position of the World Space sprites.
        /// </summary>
        private void AlignUIToWorld()
        {
            if (holeTexts == null || holeTransforms == null || holeTexts.Length != 16 || holeTransforms.Length != 16)
            {
                Debug.LogWarning("AlignUI failed: Ensure both holeTexts and holeTransforms arrays have 16 elements assigned.");
                return;
            }

            for (int i = 0; i < 16; i++)
            {
                if (holeTexts[i] != null && holeTransforms[i] != null)
                    holeTexts[i].transform.position = mainCamera.WorldToScreenPoint(holeTransforms[i].position);
            }
        }

        private void SetStatus(string msg)
        {
            Debug.Log($"[Game] {msg}");
            if (statusText != null) statusText.text = msg;
        }

        private void UpdateUI()
        {
            if (holeTexts == null || holeTexts.Length < 16) return;

            for (int i = 0; i < 16; i++)
            {
                if (holeTexts[i] != null)
                    holeTexts[i].text = game.board[i].ToString();
            }
        }

        /// <summary>
        /// Sinkronisasi jumlah objek shell visual dengan data di CongklakEngine
        /// </summary>
        private void UpdateAllHoleVisuals()
        {
            if (shellPrefab == null) return;

            for (int i = 0; i < 16; i++)
            {
                UpdateSingleHoleVisual(i);
            }
        }

        private void UpdateSingleHoleVisual(int holeIdx)
        {
            if (holeIdx < 0 || holeIdx >= holeTransforms.Length || holeTransforms[holeIdx] == null) return;
            if (holeShells == null || holeShells[holeIdx] == null) return;

            int targetCount = game.board[holeIdx];
            List<GameObject> currentShells = holeShells[holeIdx];

            // Tambah shell jika kurang
            while (currentShells.Count < targetCount)
            {
                Vector3 randomOffset = (Vector3)(Random.insideUnitCircle * holeRadius);
                Vector3 spawnPos = holeTransforms[holeIdx].position + randomOffset;
                spawnPos.z = -0.1f; // Sedikit di depan sprite lubang

                Quaternion rotation = useRandomRotation ? 
                    Quaternion.Euler(0, 0, Random.Range(0f, 360f)) : 
                    Quaternion.identity;

                GameObject newShell = Instantiate(shellPrefab, spawnPos, rotation, holeTransforms[holeIdx]);
                currentShells.Add(newShell);
            }

            // Hapus shell jika lebih (misal saat diambil atau dimakan)
            while (currentShells.Count > targetCount)
            {
                GameObject toRemove = currentShells[0];
                currentShells.RemoveAt(0);
                Destroy(toRemove);
            }
        }
    }
}
