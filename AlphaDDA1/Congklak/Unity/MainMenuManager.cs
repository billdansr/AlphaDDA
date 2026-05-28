using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.SceneManagement;

namespace CongklakAI
{
    public class MainMenuManager : MonoBehaviour
    {
        [Header("UI References")]
        public TMP_InputField nameInputField;
        public Toggle ddaToggle;
        public Button playVsAIButton;
        public Button playVsHumanButton;

        [Header("Scene Configuration")]
        public string gameSceneName = "Game"; // Nama scene game Anda

        public const string PLAYER_PREFS_NAME_KEY = "ParticipantName";
        public const string PLAYER_PREFS_DDA_KEY = "DDAEnabled";

        void Start()
        {
            // 1. Ambil nama terakhir dari PlayerPrefs (Pre-filled)
            string savedName = PlayerPrefs.GetString(PLAYER_PREFS_NAME_KEY, "");
            if (nameInputField != null)
            {
                nameInputField.text = savedName;
                // Gunakan onEndEdit agar tidak mengganggu saat sedang mengetik (mencegah flickering)
                nameInputField.onEndEdit.AddListener(OnNameValueChanged);
            }

            // 2. Ambil state DDA terakhir dari PlayerPrefs
            bool savedDDA = PlayerPrefs.GetInt(PLAYER_PREFS_DDA_KEY, 1) == 1;
            if (ddaToggle != null)
            {
                ddaToggle.isOn = savedDDA;
                ddaToggle.onValueChanged.AddListener(OnDDAToggledManually);
            }

            // 3. Daftarkan event tombol Play
            if (playVsAIButton != null)
                playVsAIButton.onClick.AddListener(() => StartGame(false)); // vs AI

            if (playVsHumanButton != null)
                playVsHumanButton.onClick.AddListener(() => StartGame(true));  // vs Human
        }

        /// <summary>
        /// Otomatis menyalakan/mematikan Toggle DDA di Main Menu saat user mengetik nama berakhiran A atau B.
        /// </summary>
        private void OnNameValueChanged(string newName)
        {
            if (ddaToggle == null) return;

            // Matikan listener sementara agar tidak mentrigger loop event
            ddaToggle.onValueChanged.RemoveListener(OnDDAToggledManually);

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

            // Daftarkan kembali listener manual setelah auto-update selesai
            ddaToggle.onValueChanged.AddListener(OnDDAToggledManually);
        }

        /// <summary>
        /// Dipanggil ketika partisipan mengubah Toggle DDA secara manual (lewat Settings di Main Menu).
        /// </summary>
        private void OnDDAToggledManually(bool isOn)
        {
            // Simpan perubahan manual segera agar persisten
            PlayerPrefs.SetInt(PLAYER_PREFS_DDA_KEY, isOn ? 1 : 0);
            PlayerPrefs.Save();
            Debug.Log($"[Main Menu] Toggle DDA diubah secara manual menjadi: {isOn}");
        }

        /// <summary>
        /// Menyimpan data dan berpindah ke scene game utama.
        /// </summary>
        public void StartGame(bool playVsHuman)
        {
            string name = "Guest";
            if (nameInputField != null && !string.IsNullOrEmpty(nameInputField.text))
            {
                name = nameInputField.text.Trim();
            }

            bool ddaEnabled = true;
            if (ddaToggle != null)
            {
                ddaEnabled = ddaToggle.isOn;
            }

            // 1. Simpan ke PlayerPrefs agar persisten di session berikutnya
            PlayerPrefs.SetString(PLAYER_PREFS_NAME_KEY, name);
            PlayerPrefs.SetInt(PLAYER_PREFS_DDA_KEY, ddaEnabled ? 1 : 0);
            PlayerPrefs.Save();

            // 2. Set static overrides ke GameController agar terbaca saat scene berpindah
            CongklakGameController.ParticipantNameOverride = name;
            CongklakGameController.IsDDAEnabledOverride = ddaEnabled;
            CongklakGameController.IsP2HumanOverride = playVsHuman;

            Debug.Log($"[Main Menu] Memulai game. Partisipan: {name} | DDA: {ddaEnabled} | vs Human: {playVsHuman}");

            // 3. Pindah Scene
            SceneManager.LoadScene(gameSceneName);
        }
    }
}
