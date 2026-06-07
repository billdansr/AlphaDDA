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
        public Toggle musicToggle;
        public Button playVsAIButton;
        public Button playVsHumanButton;

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
            Debug.Log($"[Main Menu] Musik diubah menjadi: {isOn}");
        }

        /// <summary>
        /// Menyimpan data dan berpindah ke scene game utama.
        /// </summary>
        public void StartGame(bool playVsHuman)
        {
            if (nameInputField != null)
                settings.participantName = nameInputField.text.Trim();

            settings.isP2Human = playVsHuman;
            settings.SaveToPrefs();

            SceneManager.LoadScene(gameSceneName);
        }
    }
}
