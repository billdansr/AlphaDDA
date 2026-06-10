using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.SceneManagement;

namespace CongklakAI
{
    public class MainMenuManager : MonoBehaviour
    {
        [Header("UI References")]
        public GameSettings settings; // Drag SO ke sini

        public TMP_InputField nameInputField;
        public Toggle ddaToggle;
        public Toggle consentToggle; // Toggle untuk persetujuan privasi
        public Toggle musicToggle;
        public Button playVsAIButton;
        public Button playVsHumanButton;
        public Button exitButton;
        public string privacyPolicyUrl = "https://your-website.com/privacy"; // Ganti dengan link Anda

        [Header("Scene Configuration")]
        public string gameSceneName = "Game"; // Nama scene game Anda

        void Start()
        {
            if (settings == null) return;
            settings.LoadFromPrefs();

            if (nameInputField != null)
            {
                nameInputField.text = settings.participantName;
                nameInputField.onEndEdit.AddListener(OnNameValueChanged);
            }

            if (ddaToggle != null)
            {
                ddaToggle.isOn = settings.isDda;
                ddaToggle.onValueChanged.AddListener(OnDdaToggledManually);
            }

            if (consentToggle != null)
            {
                consentToggle.isOn = settings.hasConsentedData;
                consentToggle.onValueChanged.AddListener(OnConsentToggled);
            }

            if (musicToggle != null)
            {
                musicToggle.isOn = settings.isMusicEnabled;
                musicToggle.onValueChanged.AddListener(OnMusicToggled);
            }

            // Daftarkan event tombol Play via kode (lebih aman daripada manual di Inspector)
            if (playVsAIButton != null)
                playVsAIButton.onClick.AddListener(() => StartGame(false));

            if (playVsHumanButton != null)
                playVsHumanButton.onClick.AddListener(() => StartGame(true));

            if (exitButton != null)
                exitButton.onClick.AddListener(ExitGame);
        }

        /// <summary>
        /// Otomatis menyalakan/mematikan Toggle DDA di Main Menu saat user mengetik nama berakhiran A atau B.
        /// </summary>
        private void OnNameValueChanged(string newName)
        {
            if (ddaToggle == null) return;
            
            if (newName.EndsWith("A", System.StringComparison.OrdinalIgnoreCase))
            {
                ddaToggle.isOn = false;
                Debug.Log("[Main Menu] Nama berakhiran 'A' terdeteksi. DDA otomatis Dinonaktifkan.");
            }
            else if (newName.EndsWith("B", System.StringComparison.OrdinalIgnoreCase))
            {
                ddaToggle.isOn = true;
                Debug.Log("[Main Menu] Nama berakhiran 'B' terdeteksi. DDA otomatis Diaktifkan.");
            }
        }

        /// <summary>
        /// Dipanggil ketika partisipan mengubah Toggle DDA secara manual (lewat Settings di Main Menu).
        /// </summary>
        private void OnDdaToggledManually(bool isOn)
        {
            settings.isDda = isOn;
            settings.SaveToPrefs();
            Debug.Log($"[Main Menu] Toggle DDA diubah secara manual menjadi: {isOn}");
        }

        /// <summary>
        /// Dipanggil ketika partisipan mengubah Toggle Musik secara manual.
        /// </summary>
        private void OnMusicToggled(bool isOn)
        {
            settings.isMusicEnabled = isOn;
            settings.SaveToPrefs();
            GlobalMusicPlayer.Refresh();
            Debug.Log($"[Main Menu] Musik diubah menjadi: {isOn}");
        }

        private void OnConsentToggled(bool isOn)
        {
            settings.hasConsentedData = isOn;
            settings.SaveToPrefs();
            Debug.Log($"[Main Menu] Persetujuan data: {isOn}");
        }

        /// <summary>
        /// Membuka URL Kebijakan Privasi di browser (Wajib untuk Play Store).
        /// Panggil fungsi ini dari Button "Kebijakan Privasi".
        /// </summary>
        public void OpenPrivacyPolicy()
        {
            Application.OpenURL(privacyPolicyUrl);
        }

        /// <summary>
        /// Menyimpan data dan berpindah ke scene game utama.
        /// </summary>
        public void StartGame(bool playVsHuman)
        {
            string enteredName = nameInputField != null ? nameInputField.text.Trim() : "";

            // Logika Deteksi Partisipan Baru:
            // Jika nama yang dimasukkan berbeda dengan yang tersimpan, reset counter sesi.
            if (!string.IsNullOrEmpty(enteredName) && 
                !string.Equals(enteredName, settings.participantName, System.StringComparison.OrdinalIgnoreCase))
            {
                settings.sessionCount = 0;
                Debug.Log($"[Main Menu] Partisipan baru '{enteredName}' dideteksi. Mengatur ulang sessionCount ke 0.");
            }
            
            settings.participantName = enteredName;

            // Validasi: Jika data mau dikirim ke Google Form, pastikan sudah setuju
            if (!settings.hasConsentedData && !playVsHuman)
            {
                // Anda bisa menambahkan popup peringatan di sini
                Debug.LogWarning("Persetujuan data diperlukan untuk mengirim log penelitian.");
            }

            settings.isP2Human = playVsHuman;
            settings.SaveToPrefs();

            SceneManager.LoadScene(gameSceneName);
        }

        /// <summary>
        /// Fungsi manual untuk me-reset data partisipan (bisa dihubungkan ke Button "Partisipan Baru").
        /// </summary>
        public void ResetParticipantSession()
        {
            settings.ResetSessionCount();
            if (nameInputField != null) nameInputField.text = "";
            settings.participantName = "";
            settings.SaveToPrefs();
        }

        /// <summary>
        /// Menutup aplikasi.
        /// </summary>
        public void ExitGame()
        {
            Debug.Log("[Main Menu] Keluar dari permainan...");
            #if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
            #else
                Application.Quit();
            #endif
        }
    }
}
