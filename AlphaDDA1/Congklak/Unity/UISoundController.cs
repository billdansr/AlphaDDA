using UnityEngine;
using UnityEngine.UI;

namespace CongklakAI
{
    /// <summary>
    /// Komponen otomatis untuk memutar suara klik dari GameSettings.
    /// Bisa ditempelkan ke Button atau Toggle.
    /// </summary>
    public class UISoundController : MonoBehaviour
    {
        public GameSettings settings;
        private AudioSource globalSource;

        void Start()
        {
            // Otomatis memuat GameSettings jika belum di-assign di Inspector
            if (settings == null)
            {
                settings = Resources.Load<GameSettings>("GameSettings");
            }

            if (settings == null) return;

            // Cari atau buat AudioSource untuk memutar suara UI
            // Kita gunakan AudioSource global agar suara tidak terputus jika objek hancur
            globalSource = GameObject.Find("UIAudioSource")?.GetComponent<AudioSource>();
            if (globalSource == null)
            {
                GameObject go = new GameObject("UIAudioSource");
                globalSource = go.AddComponent<AudioSource>();
                DontDestroyOnLoad(go); // Agar suara tetap bunyi saat pindah scene
            }

            // Daftarkan listener secara otomatis berdasarkan tipe komponen
            Button btn = GetComponent<Button>();
            if (btn != null)
            {
                // Menggunakan lambda agar aman jika method diubah
                btn.onClick.AddListener(() => PlayClick());
            }

            Toggle tgl = GetComponent<Toggle>();
            if (tgl != null)
            {
                tgl.onValueChanged.AddListener((_) => PlayClick());
            }
        }

        public void PlayClick()
        {
            // Menambahkan cek isMusicEnabled agar suara SFX mengikuti setting global jika diinginkan
            if (globalSource != null && settings != null && settings.buttonClickSound != null && settings.isMusicEnabled)
            {
                globalSource.volume = settings.sfxVolume;
                globalSource.PlayOneShot(settings.buttonClickSound);
            }
        }
    }
}