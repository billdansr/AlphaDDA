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
                ddaToggle.isOn = settings.isDDAEnabled;
                ddaToggle.onValueChanged.AddListener(OnDDAToggledManually);
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
        private void OnDDAToggledManually(bool isOn)
        {
            settings.isDDAEnabled = isOn;
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
            if (nameInputField != null)
                settings.participantName = nameInputField.text.Trim();

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
    }
}
