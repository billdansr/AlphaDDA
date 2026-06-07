using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Threading.Tasks;
using UnityEngine.InputSystem;
using TMPro;
using UnityEngine.SceneManagement;

namespace CongklakAI
{
    public class CongklakGameController : MonoBehaviour
    {
        #region 1. Configuration & Parameters
        public GameSettings settings; // Drag SO ke sini

        [Header("AI Configuration")]
        public AIBrain aiBrain;
        public bool isP1Human = true;
        public bool isP2Human = false;

        [Header("Difficulty Parameters")]
        public float sensitivityA = 1.0f; // Calibrated Golden Parameter (A=1.0)
        public float offsetX0 = -2.0f;    // Calibrated Golden Parameter (X0=-2.0)
        public int maxSims = 300;
        public float stepDelay = 0.2f; // Time between shell drops
        #endregion

        [Header("Visual Tuning")]
        public float holeRadius = 0.15f; // Radius penyebaran biji di dalam lubang
        public float storeRadius = 0.35f; // Radius lebih luas khusus untuk lumbung (Store)
        public bool useRandomRotation = true;

        [Header("UI Offset Settings")]
        public float uiVerticalOffset = 1.2f;   // Jarak atas/bawah
        public float uiHorizontalOffset = 1.5f; // Jarak kiri/kanan

        
        

        [Header("Animation Settings")]
        public GameObject shellPrefab;   // Assign a small shell/circle prefab
        public float shellMoveSpeed = 15f; // Speed of the shell moving between holes
        public float handTravelDuration = 0.5f; // Durasi gerakan tangan naik/turun (Hole <-> Hand)

        public string participantName = "Guest";

        [Header("DDA Settings")]
        public bool isDDAEnabled = true;

        [Header("Google Form Configuration")]
        public string gFormUrl = "https://docs.google.com/forms/d/e/.../formResponse";
        public string[] entryIds = new string[11]; // Fill with Google Form entry IDs (e.g., "entry.123456"). See gform_setup.md for guidance.

        private float gameStartTime;
        private float turnStartTime;
        private List<string> gameLogs = new List<string>();        

        public Transform p1HandTarget;    // Titik di dekat area bawah (P1)
        public Transform p2HandTarget;    // Titik di dekat area atas (P2)
        
        [Header("Game Over UI Popup")]
        public GameObject gameOverPanel;               // Drag Game Over Panel here
        public TMPro.TextMeshProUGUI gameOverWinnerText; // Drag TextMeshPro text for winner here
        public TMPro.TextMeshProUGUI gameOverDetailsText; // Drag TextMeshPro text for scores/stats here
        public TMPro.TextMeshProUGUI uploadStatusText;   // Drag TextMeshPro text for upload status here
        public UnityEngine.UI.Button playAgainButton;    // Drag Play Again button here
        public UnityEngine.UI.Button mainMenuButton;     // Drag Main Menu button here
        public string mainMenuSceneName = "Main Menu";  // Avoid magic strings! Expose scene name in inspector

        [Header("Audio Settings")]
        public AudioSource audioSource;
        public AudioSource musicSource;
        public AudioClip swooshSound;
        public AudioClip dropSound;
        public AudioClip bgMusic;
        public AudioClip victorySound;
        public AudioClip defeatSound;

        [Header("Glow Aesthetics")]
        public Color p1GlowColor = Color.red;
        public Color p2GlowColor = Color.blue;
        public float glowBaseScale = 0.15f; // Skala dasar agar pas dengan lubang
        public float glowPulseSpeed = 5f;
        public float glowPulseAmount = 0.15f; 
        public float tapScaleFactor = 0.8f; // Skala saat ditekan (Tap Feedback)

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
        private GameObject[] selectionGlows;
        private bool isSyncingLogs = false;
        private Coroutine statusAnimationCoroutine;
        private Coroutine handTextPunchCoroutine;

        void Start()
        {
            // Seed true randomness using system ticks to prevent repetitive Editor patterns
            UnityEngine.Random.InitState((int)System.DateTime.Now.Ticks);

            if (gameOverPanel != null) gameOverPanel.SetActive(false);
            
            // Lokalisasi teks tombol secara dinamis untuk konsistensi bahasa
            if (playAgainButton != null)
            {
                var btnText = playAgainButton.GetComponentInChildren<TMP_Text>();
                if (btnText != null) btnText.text = "Main Lagi";
            }

            if (mainMenuButton != null)
            {
                var btnText = mainMenuButton.GetComponentInChildren<TMP_Text>();
                if (btnText != null) btnText.text = "Menu Utama";
            }

            // Sinkronisasi data dari Singleton GameSettings
            if (settings == null) return;
            participantName = string.IsNullOrEmpty(settings.participantName) 
                ? "Guest" 
                : settings.participantName;
            isDDAEnabled = settings.isDDAEnabled;
            isP2Human = settings.isP2Human;
            
            if (audioSource != null) audioSource.volume = settings.sfxVolume;
            if (musicSource != null) musicSource.volume = settings.musicVolume;

            // Clear DDA's persistent win score queue from previous games/sessions
            CongklakAI.AlphaDDA_MCTS.ResetDDA();

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
            
            if (handShellCounterText != null)
            {
                handShellCounterText.gameObject.SetActive(false); // Hide initially
            }

            // Inisialisasi Selection Glows dari Child (Mobile First)
            selectionGlows = new GameObject[16];
            for (int i = 0; i < 16; i++)
            {
                if (holeTransforms[i] != null)
                {
                    Transform glowT = holeTransforms[i].Find("Selection_Glow");
                    if (glowT != null) selectionGlows[i] = glowT.gameObject;
                }
            }

            // Kontrol Eksperimen: Manusia selalu P1 untuk menstandarisasi giliran pertama.
            // Ini mencegah 'first-mover advantage' menjadi variabel pengganggu dalam analisis DDA.
            isP1Human = true;

            // Mulai pencatatan waktu sesi
            gameStartTime = Time.time;
            turnStartTime = Time.time;

            StartCoroutine(GameLoop());

            // Coba sinkronisasi data lama yang mungkin gagal kirim di sesi sebelumnya
            StartCoroutine(UploadPendingLogs());
        }

        

        private void LogMove(int player, int move, float v, int sims, int s1, int s2)
        {
            float thinkTime = Time.time - turnStartTime;
            string logRow = $"{participantName},{isDDAEnabled},{Time.time - gameStartTime:F2},{turnCount},{player},{move},{v:F3},{sims},{thinkTime:F2},{s1},{s2}";
            gameLogs.Add(logRow);
            
            // Reset timer untuk langkah berikutnya
            turnStartTime = Time.time;
        }

        private void ExportLogsToCSV()
        {
            // 1. Simpan ke folder 'Pending'
            string folderPath = System.IO.Path.Combine(Application.persistentDataPath, "Logs", "Pending");
            if (!System.IO.Directory.Exists(folderPath)) System.IO.Directory.CreateDirectory(folderPath);

            string fileName = $"GameLog_{participantName}_{System.DateTime.Now:yyyyMMdd_HHmmss}.csv";
            string filePath = System.IO.Path.Combine(folderPath, fileName);

            string header = "Participant,IsDDAEnabled,TimeInSession,Turn,Player,Move,BoardEval_V,Simulations,ThinkTime,ScoreP1,ScoreP2";
            string content = header + string.Join("\n", gameLogs);

            System.IO.File.WriteAllText(filePath, content);
            Debug.Log($"[Logger] Log disimpan ke folder Pending: {filePath}");

            // 2. Jalankan proses sinkronisasi
            StartCoroutine(UploadPendingLogs());
        }

        private IEnumerator UploadPendingLogs()
        {
            if (isSyncingLogs) yield break;
            isSyncingLogs = true;

            string pendingPath = System.IO.Path.Combine(Application.persistentDataPath, "Logs", "Pending");
            string uploadedPath = System.IO.Path.Combine(Application.persistentDataPath, "Logs", "Uploaded");

            if (!System.IO.Directory.Exists(pendingPath)) { isSyncingLogs = false; yield break; }
            if (!System.IO.Directory.Exists(uploadedPath)) System.IO.Directory.CreateDirectory(uploadedPath);

            string[] pendingFiles = System.IO.Directory.GetFiles(pendingPath, "*.csv");
            if (pendingFiles.Length == 0) { isSyncingLogs = false; yield break; }

            Debug.Log("[Logger] Memulai upload data ke Google Sheets...");
            
            foreach (string file in pendingFiles)
            {
                string[] lines = System.IO.File.ReadAllLines(file);
                int lineCount = 0;

                foreach (string line in lines)
                {
                    if (string.IsNullOrWhiteSpace(line) || line.StartsWith("Participant")) continue;

                    string[] data = line.Split(',');
                    yield return StartCoroutine(SendRowToGoogle(data));
                    
                    lineCount++;
                    if (uploadStatusText != null) 
                        uploadStatusText.text = $"Sinkronisasi ({pendingFiles.Length} file): {lineCount}/{lines.Length - 1}";
                }

                // Pindahkan ke folder 'Uploaded' hanya jika file selesai diproses
                string destFile = System.IO.Path.Combine(uploadedPath, System.IO.Path.GetFileName(file));
                if (System.IO.File.Exists(destFile)) System.IO.File.Delete(destFile);
                System.IO.File.Move(file, destFile);
            }

            if (uploadStatusText != null) uploadStatusText.text = "Semua data berhasil disinkronkan!";
            
            // Jika masih ada file yang tersisa di folder pending (karena gagal upload di tengah jalan)
            int remaining = System.IO.Directory.GetFiles(pendingPath, "*.csv").Length;
            if (remaining > 0 && uploadStatusText != null)
            {
                uploadStatusText.text = $"Sinkronisasi terhenti. {remaining} sesi tersimpan lokal (Offline).";
            }
            
            // Aktifkan kembali tombol navigasi
            if (playAgainButton != null) playAgainButton.interactable = true;
            if (mainMenuButton != null) mainMenuButton.interactable = true;
            
            isSyncingLogs = false;
        }

        private IEnumerator SendRowToGoogle(string[] data)
        {
            if (string.IsNullOrEmpty(gFormUrl) || entryIds == null || entryIds.Length == 0) yield break;

            WWWForm form = new WWWForm();
            for (int i = 0; i < entryIds.Length && i < data.Length; i++)
            {
                if (!string.IsNullOrEmpty(entryIds[i]))
                {
                    form.AddField(entryIds[i], data[i]);
                }
            }

            using (UnityEngine.Networking.UnityWebRequest www = UnityEngine.Networking.UnityWebRequest.Post(gFormUrl, form))
            {
                yield return www.SendWebRequest();

                if (www.result != UnityEngine.Networking.UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning("[Logger] Gagal upload baris: " + www.error);
                }
            }
        }

        /// <summary>
        /// Tombol Rahasia: Mengabaikan setelan nama dan mengubah status DDA secara manual saat game berlangsung.
        /// </summary>
        public void ToggleDDAManually()
        {
            isDDAEnabled = !isDDAEnabled;
            settings.isDDAEnabled = isDDAEnabled;
            settings.SaveToPrefs();
            Debug.Log($"[Logger] DDA secara manual diubah menjadi: {isDDAEnabled}");
            SetStatus($"DDA {(isDDAEnabled ? "AKTIF" : "NON-AKTIF")} (Manual)");
        }

        /// <summary>
        /// Dipanggil langsung oleh UI Toggle di dalam game (Event: On Value Changed).
        /// </summary>
        /// <param name="isEnabled">Status toggle DDA</param>
        public void SetDDAGameToggle(bool isEnabled)
        {
            isDDAEnabled = isEnabled;
            settings.isDDAEnabled = isEnabled;
            settings.SaveToPrefs();
            Debug.Log($"[Settings] DDA diubah lewat UI menjadi: {isEnabled}");
            SetStatus($"DDA {(isEnabled ? "AKTIF" : "NON-AKTIF")}");
        }
        


        void Update()
        {
            // Fallback jika kamera belum ter-assign (misal saat ganti scene)
            if (mainCamera == null) mainCamera = Camera.main;

            // Selalu sinkronkan posisi teks UI dengan posisi lubang
            AlignUIToWorld();

            // Update Highlight (Glow) untuk lubang yang aktif
            UpdateHighlights();

            if (mainCamera == null || !isInteracting) return;

            // Detect clicks on World Space Sprites using the New Input System
            if (Pointer.current != null && Pointer.current.press.wasReleasedThisFrame)
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
                bool isHuman = (game.currentPlayer == 1 && isP1Human) || (game.currentPlayer == -1 && isP2Human);
                string playerLabel = GetFormattedPlayerLabel(game.currentPlayer);
                
                // Cek apakah pemain saat ini punya langkah. Jika tidak, lompati giliran (Pass)
                if (game.GetValidMoves().Count == 0)
                {
                    SetStatus($"{playerLabel} kosong, melompati giliran...");
                    game.currentPlayer *= -1;
                    yield return new WaitForSeconds(1.0f);
                    continue;
                }

                int playerTag = game.currentPlayer;

                if (isHuman)
                {
                    SetStatus($"Giliran {playerLabel}");
                    pendingMove = -1;
                    isInteracting = true;
                    yield return new WaitUntil(() => pendingMove != -1);
                    
                    // LOG HUMAN MOVE
                    LogMove(playerTag, pendingMove, 0, 0, game.board[7], game.board[15]);
                    
                    SetStatus($"{playerLabel} Berjalan");
                    yield return StartCoroutine(ExecuteMove(pendingMove));
                }
                else
                {
                    SetStatus($"{playerLabel} Berpikir..."); // Log awal
                    
                    // Mulai animasi titik-titik
                    if (statusAnimationCoroutine != null) StopCoroutine(statusAnimationCoroutine);
                    statusAnimationCoroutine = StartCoroutine(AnimateThinkingStatus($"{playerLabel} Berpikir"));

                    int aiMove = -1;
                    float activeSensitivity = isDDAEnabled ? sensitivityA : 0.0f;
                    AlphaDDA_MCTS mcts = new AlphaDDA_MCTS(game, aiBrain, activeSensitivity, offsetX0, maxSims);
                    
                    // Run MCTS on main thread via Coroutine (Inference safe)
                    yield return StartCoroutine(mcts.RunCoroutine(turnCount, (move) => aiMove = move));

                    // Hentikan animasi setelah berpikir selesai
                    if (statusAnimationCoroutine != null) StopCoroutine(statusAnimationCoroutine);
                    statusAnimationCoroutine = null;

                    // LOG AI MOVE (Capture DDA Metrics)
                    LogMove(playerTag, aiMove, mcts.lastV, mcts.lastSims, game.board[7], game.board[15]);

                    yield return new WaitForSeconds(0.5f); // Cosmetic delay
                    if (aiMove != -1)
                    {
                        SetStatus($"{playerLabel} Berjalan");
                        yield return StartCoroutine(ExecuteMove(aiMove));
                    }
                    else
                        Debug.LogError("AI failed to return a valid move!");
                }

                yield return null;
            }

            string winnerLabel = "Seri";
            if (game.winner != 0)
                winnerLabel = GetFormattedPlayerLabel(game.winner);
            
            SetStatus($"Permainan Selesai! Pemenang: {winnerLabel}");
            Debug.Log($"[Game] Game Over! Winner: P{(game.winner == 1 ? "1" : "2")}");
            
            // 2. Tentukan Pemenang dan mainkan SFX kemenangan/kekalahan
            bool humanWon = (game.winner == 1 && isP1Human) || (game.winner == -1 && isP2Human);
            if (audioSource != null)
            {
                AudioClip clip = humanWon ? victorySound : defeatSound;
                if (clip != null) audioSource.PlayOneShot(clip);
            }

            // 3. Tampilkan UI Panel Game Over
            if (gameOverPanel != null)
            {
                gameOverPanel.SetActive(true);
                
                if (gameOverWinnerText != null)
                {
                    gameOverWinnerText.text = humanWon ? "Mantap! Kamu Juaranya! 🎉" : "Yah, AI lebih jago kali ini! Semangat!";
                }
                
                if (gameOverDetailsText != null)
                {
                    // P1 selalu Human (Opsi A), P2 selalu AI
                    gameOverDetailsText.text = $"Skor Akhir\nP1: {game.board[7]}  |  P2: {game.board[15]}\nTotal Giliran: {turnCount}\nKeterangan: Kamu jalan pertama (P1)";
                }
                
                // Matikan tombol sementara agar data penelitian terupload aman ke Google Form!
                if (playAgainButton != null) playAgainButton.interactable = false;
                if (mainMenuButton != null) mainMenuButton.interactable = false;
                if (uploadStatusText != null) uploadStatusText.text = "Menyimpan data penelitian ke Cloud, mohon tunggu...";
            }

            // EXPORT DATA SETELAH SELESAI
            ExportLogsToCSV();
        }

        /// <summary>
        /// Dipanggil ketika tombol Play Again di Panel Game Over diklik.
        /// </summary>
        public void RestartGame()
        {
            settings.FlipDDA();
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        }

        /// <summary>
        /// Dipanggil ketika tombol Main Menu di Panel Game Over diklik.
        /// </summary>
        public void BackToMainMenu()
        {
            settings.FlipDDA();
            settings.SaveToPrefs();

            SceneManager.LoadScene(mainMenuSceneName);
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
                handShellCounterText.color = (game.currentPlayer == 1) ? p1GlowColor : p2GlowColor;
                handShellCounterText.text = initialShells.ToString();
                handShellCounterText.gameObject.SetActive(true);

                if (handTextPunchCoroutine != null) StopCoroutine(handTextPunchCoroutine);
                handTextPunchCoroutine = StartCoroutine(AnimatePunchScale(handShellCounterText.transform));
            }

            yield return StartCoroutine(AnimatePickup(startHole, handShell, true));
            bool isFirstDropAfterPickup = true; // Flag untuk gerakan pertama

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
                        isFirstDropAfterPickup = true; // Set ulang flag karena baru saja ambil biji lagi
                        UpdateAllHoleVisuals();
                        UpdateUI();
                        
                        // Update Hand Text SETELAH ambil biji (Jalan Terus)
                        if (handShellCounterText != null)
                        {
                            handShellCounterText.text = remainingShells.ToString();
                            if (handTextPunchCoroutine != null) StopCoroutine(handTextPunchCoroutine);
                            handTextPunchCoroutine = StartCoroutine(AnimatePunchScale(handShellCounterText.transform));
                        }
                    }

                    lastStoreCount = game.board[storeIdx];
                    continue;
                }
                Vector3 targetPos = holeTransforms[holeIdx].position;
                targetPos.z = zOffset;
                
                Vector3 startPos = handShell.transform.position;
                float distance = Vector3.Distance(startPos, targetPos);
                
                // Gunakan durasi tetap untuk drop pertama (simetris dengan pickup), dan kecepatan linear untuk drop selanjutnya
                float duration = isFirstDropAfterPickup ? handTravelDuration : (distance / shellMoveSpeed);
                isFirstDropAfterPickup = false; // Reset flag setelah drop pertama dilakukan
                
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
                    if (handTextPunchCoroutine != null) StopCoroutine(handTextPunchCoroutine);
                    handTextPunchCoroutine = StartCoroutine(AnimatePunchScale(handShellCounterText.transform));
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
            string playerLabel = GetFormattedPlayerLabel(game.currentPlayer);
            SetStatus($"{playerLabel} TEMBAK!");
            
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
            float duration = handTravelDuration;
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
                string playerLabel = GetFormattedPlayerLabel(game.currentPlayer);
                SetStatus($"{playerLabel} JALAN TERUS!");
                
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
            float duration = handTravelDuration;
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
            if (statusText != null)
            {
                statusText.text = msg;
                statusText.color = Color.white; // Reset warna dasar agar Rich Text bekerja
            }
            Debug.Log($"[Game] {msg}");
        }

        /// <summary>
        /// Animasi teks titik-titik (loading) agar UI terasa hidup saat proses berat.
        /// </summary>
        private IEnumerator AnimateThinkingStatus(string baseText)
        {
            // Menggunakan Rich Text tag <color=#00000000> (alpha 0) 
            // Agar lebar teks "Thinking..." selalu tetap 3 titik.
            // Ini mencegah teks bergoyang/geser saat alignment-nya Center.
            string[] dotSequences = { 
                ".<color=#00000000>..</color>", 
                "..<color=#00000000>.</color>", 
                "..." 
            };
            int dotCount = 1;
            
            while (true)
            {
                if (statusText != null) 
                    statusText.text = baseText + dotSequences[dotCount - 1];
                
                dotCount = (dotCount % 3) + 1;
                yield return new WaitForSeconds(0.4f);
            }
        }

        /// <summary>
        /// Memberikan efek 'Juice' berupa perubahan skala mendadak pada teks UI.
        /// </summary>
        private IEnumerator AnimatePunchScale(Transform target)
        {
            float duration = 0.15f;
            Vector3 punchScale = Vector3.one * 1.4f;
            float elapsed = 0;

            // Scale Up (Cepat)
            while (elapsed < duration * 0.3f)
            {
                target.localScale = Vector3.Lerp(Vector3.one, punchScale, elapsed / (duration * 0.3f));
                elapsed += Time.deltaTime;
                yield return null;
            }
            // Scale Down (Meredam)
            elapsed = 0;
            while (elapsed < duration * 0.7f)
            {
                target.localScale = Vector3.Lerp(punchScale, Vector3.one, elapsed / (duration * 0.7f));
                elapsed += Time.deltaTime;
                yield return null;
            }
            target.localScale = Vector3.one;
            handTextPunchCoroutine = null;
        }

        /// <summary>
        /// Membuat label pemain dengan pewarnaan Rich Text khusus pada bagian "P1" atau "P2".
        /// </summary>
        private string GetFormattedPlayerLabel(int player)
        {
            Color col = (player == 1) ? p1GlowColor : p2GlowColor;
            string hex = ColorUtility.ToHtmlStringRGB(col);
            return $"<color=#{hex}>P{(player == 1 ? "1" : "2")}</color>";
        }

        private void UpdateHighlights()
        {
            if (selectionGlows == null) return;

            // Hitung denyut berdasarkan waktu
            float pulse = 1f + Mathf.Sin(Time.time * glowPulseSpeed) * glowPulseAmount;

            // Deteksi posisi sentuhan untuk Tap Feedback (Mobile First)
            Vector2 pointerPos = Pointer.current.position.ReadValue();
            float zDist = Mathf.Abs(mainCamera.transform.position.z);
            Vector3 worldPos3D = mainCamera.ScreenToWorldPoint(new Vector3(pointerPos.x, pointerPos.y, zDist));
            Vector2 worldPos2D = new Vector2(worldPos3D.x, worldPos3D.y);
            Collider2D pressedCollider = (Pointer.current != null && Pointer.current.press.isPressed) ? Physics2D.OverlapPoint(worldPos2D) : null;

            for (int i = 0; i < 16; i++)
            {
                if (selectionGlows[i] == null) continue;

                // Tampilkan glow jika: 1. Giliran manusia, 2. Lubang milik pemain aktif, 3. Lubang ada isinya
                bool isP1Hole = (i >= 0 && i <= 6);
                bool shouldShow = isInteracting && IsMyTurnHole(i) && game.board[i] > 0;
                
                if (selectionGlows[i].activeSelf != shouldShow)
                    selectionGlows[i].SetActive(shouldShow);

                // Efek visual saat aktif
                if (shouldShow)
                {
                    float currentPulse = pulse;

                    // Tap Feedback: Mengecilkan skala jika sedang ditekan
                    if (pressedCollider != null && pressedCollider.gameObject == holeTransforms[i].gameObject)
                    {
                        currentPulse *= tapScaleFactor;
                    }

                    // 1. Terapkan Denyut (Pulse) relatif terhadap skala dasar
                    selectionGlows[i].transform.localScale = Vector3.one * (glowBaseScale * currentPulse);

                    // 2. Terapkan Rotasi
                    selectionGlows[i].transform.Rotate(0, 0, Time.deltaTime * 60f);

                    // 3. Terapkan Warna berdasarkan Pemain
                    if (selectionGlows[i].TryGetComponent<SpriteRenderer>(out var sr))
                    {
                        sr.color = isP1Hole ? p1GlowColor : p2GlowColor;
                    }
                }
            }
        }

        private bool IsMyTurnHole(int idx)
        {
            if (game.currentPlayer == 1) return idx >= 0 && idx <= 6;
            return idx >= 8 && idx <= 14;
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

            // Gunakan radius yang berbeda jika ini adalah Store (Lumbung)
            float r = (holeIdx == 7 || holeIdx == 15) ? storeRadius : holeRadius;

            // Tambah shell jika kurang
            while (currentShells.Count < targetCount)
            {
                Vector3 randomOffset = (Vector3)(Random.insideUnitCircle * r);
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
