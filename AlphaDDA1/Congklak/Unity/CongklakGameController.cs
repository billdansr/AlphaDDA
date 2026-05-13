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
        public float sensitivityA = 10.0f; // Default disamakan dengan AlphaDDA1.py
        public float offsetX0 = 0.0f;
        public int maxSims = 300;
        public float stepDelay = 0.2f; // Time between shell drops

        [Header("Visual Tuning")]
        public float holeRadius = 0.15f; // Radius penyebaran biji di dalam lubang
        public bool useRandomRotation = true;

        [Header("UI Offset Settings")]
        public float uiVerticalOffset = 1.2f;   // Jarak atas/bawah
        public float uiHorizontalOffset = 1.5f; // Jarak kiri/kanan

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
            // Fallback jika kamera belum ter-assign (misal saat ganti scene)
            if (mainCamera == null) mainCamera = Camera.main;

            // Selalu sinkronkan posisi teks UI dengan posisi lubang
            AlignUIToWorld();

            if (mainCamera == null || !isInteracting) return;

            // Detect clicks on World Space Sprites using the New Input System
            if (Pointer.current != null && Pointer.current.press.wasPressedThisFrame)
            {
                Vector3 mousePos = Pointer.current.position.ReadValue();

                // Hitung jarak Z dari kamera ke bidang papan (Z=0).
                // Ini membuat kode bekerja baik di mode Perspective maupun Orthographic.
                float zDist = Mathf.Abs(mainCamera.transform.position.z);
                Vector3 worldPos3D = mainCamera.ScreenToWorldPoint(new Vector3(mousePos.x, mousePos.y, zDist));
                Vector2 worldPosition = new Vector2(worldPos3D.x, worldPos3D.y);

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
            float zOffset = -0.5f; 
            int lastHoleIdx = -1;
            int storeIdx = (game.currentPlayer == 1) ? 7 : 15;
            int lastStoreCount = game.board[storeIdx];

            // 1. Initial Pickup: Ambil dari lubang awal (Start Hole)
            int startHole = (game.currentPlayer == 1) ? move : move + 8;
            int initialShells = game.board[startHole];

            handShell = new GameObject("HandGroup");
            handShell.transform.position = holeTransforms[startHole].position;

            if (handShellCounterText != null)
            {
                handShellCounterText.text = initialShells.ToString();
                handShellCounterText.gameObject.SetActive(true);
            }

            yield return StartCoroutine(AnimatePickup(startHole, handShell, true));
            // HAPUS: UpdateAllHoleVisuals() di sini menyebabkan biji muncul lagi karena 
            // engine belum dipanggil (board[startHole] masih berisi).

            // 2. Main Loop: Distribusi Bidak
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
                    if (game.board[storeIdx] > lastStoreCount)
                    {
                        yield return StartCoroutine(AnimateCapture(holeIdx, handShell));
                        handShell = null;
                        if (handShellCounterText != null) handShellCounterText.gameObject.SetActive(false);
                    }
                    else
                    {
                        yield return StartCoroutine(AnimatePickup(holeIdx, handShell, false));
                        UpdateAllHoleVisuals();
                        UpdateUI();
                        
                        // Update Hand Text SETELAH ambil biji (Jalan Terus)
                        if (handShellCounterText != null)
                        {
                            handShellCounterText.text = remainingShells.ToString();
                        }
                    }

                    lastStoreCount = game.board[storeIdx];
                    continue;
                }
                Vector3 targetPos = holeTransforms[holeIdx].position;
                targetPos.z += zOffset;
                
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

                // Sinkronisasi Visual Drop: Kurangi prefab di tangan agar sesuai data engine
                if (handShell.transform.childCount > remainingShells)
                {
                    int toDestroy = handShell.transform.childCount - remainingShells;
                    for (int i = 0; i < toDestroy; i++)
                    {
                        if (handShell.transform.childCount > 0)
                            Destroy(handShell.transform.GetChild(0).gameObject);
                    }
                }
                
                if (audioSource != null && dropSound != null)
                    audioSource.PlayOneShot(dropSound);

                UpdateAllHoleVisuals();
                UpdateUI();

                // Update Hand Text SETELAH sinkronisasi papan agar sinkron secara visual
                if (handShellCounterText != null)
                {
                    handShellCounterText.text = remainingShells.ToString();
                }

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

            // 1. Ambil semua shell yang akan dipindahkan ke lumbung
            List<GameObject> capturedObjects = new List<GameObject>();

            // Ambil dari lubang lawan (opposite)
            if (holeShells[oppositeIdx].Count > 0)
            {
                capturedObjects.AddRange(holeShells[oppositeIdx]);
                holeShells[oppositeIdx].Clear();
            }

            // Ambil dari lubang sendiri (yang baru saja diisi)
            if (holeShells[landingHoleIdx].Count > 0)
            {
                capturedObjects.AddRange(holeShells[landingHoleIdx]);
                holeShells[landingHoleIdx].Clear();
            }
            
            // Tambahkan shell sisa dari visual tangan (jika ada)
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
                
                // Berikan posisi acak lokal yang sangat kecil agar terlihat saling menyentuh di tangan
                obj.transform.localPosition = (Vector3)(Random.insideUnitCircle * 0.05f);
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
                return;
            }

            for (int i = 0; i < 16; i++)
            {
                if (holeTexts[i] != null && holeTransforms[i] != null)
                {
                    Vector3 worldPos = holeTransforms[i].position;

                    // Terapkan Offset berdasarkan posisi lubang agar tidak menutupi biji (kewuk)
                    if (i >= 0 && i <= 6) // Lubang Bawah (P1)
                        worldPos += Vector3.down * uiVerticalOffset;
                    else if (i == 7) // Store P1
                        worldPos += Vector3.left * uiHorizontalOffset;
                    else if (i >= 8 && i <= 14) // Lubang Atas (P2)
                        worldPos += Vector3.up * uiVerticalOffset;
                    else if (i == 15) // Store P2
                        worldPos += Vector3.right * uiHorizontalOffset;

                    holeTexts[i].transform.position = mainCamera.WorldToScreenPoint(worldPos);
                }
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
