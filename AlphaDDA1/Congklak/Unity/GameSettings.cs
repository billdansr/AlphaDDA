using UnityEngine;

namespace CongklakAI
{
    [CreateAssetMenu(fileName = "GameSettings", menuName = "Congklak/GameSettings")]
    public class GameSettings : ScriptableObject
    {
        [Header("Persistent Data")]
        public string participantName = "";
        public bool isDDAEnabled = true;
        public bool isMusicEnabled = true;
        public float musicVolume = 0.4f; // Default lebih pelan
        public float sfxVolume = 1.0f;   // Default maksimal
        public bool isP2Human = false;

        [Header("UI Terminology")]
        public string termTurn = "Giliran";
        public string termSkip = "kosong, melompati giliran...";
        public string termMoving = "Berjalan";
        public string termCapture = "Makan!";
        public string termContinue = "Jalan Terus!";
        public string termThinking = "Berpikir";
        public string termGameOver = "Permainan Selesai! Pemenang:";
        public string termVictory = "Selamat! Anda Menang!";
        public string termDefeat = "Permainan Selesai. Kemenangan untuk AI.";
        public string termDraw = "Remis";
        public string termFinalScore = "Skor Akhir";
        public string termTotalTurns = "Total Giliran";
        public string termP1FirstInfo = "Keterangan: Kamu jalan pertama (P1)";
        public string termSyncing = "Menyimpan data penelitian ke Cloud, mohon tunggu...";
        public string termSyncDone = "Semua data berhasil disinkronkan!";
        public string termSyncOffline = "Sinkronisasi terhenti. Sesi tersimpan lokal (Offline).";
        public string termPlayAgain = "Main Lagi";
        public string termMainMenu = "Menu Utama";

        [Header("Global Audio Assets")]
        public AudioClip backgroundMusic;
        public AudioClip buttonClickSound;

        [Header("Gameplay SFX Assets")]
        public AudioClip swooshSound;
        public AudioClip dropSound;
        public AudioClip victorySound;
        public AudioClip defeatSound;

        public const string PREF_NAME = "Name";
        public const string PREF_DDA = "DDA";
        public const string PREF_MUSIC = "Music";
        public const string PREF_MUSIC_VOL = "MusicVolume";
        public const string PREF_SFX_VOL = "SFXVolume";

        public void SaveToPrefs()
        {
            PlayerPrefs.SetString(PREF_NAME, participantName);
            PlayerPrefs.SetInt(PREF_DDA, isDDAEnabled ? 1 : 0);
            PlayerPrefs.SetInt(PREF_MUSIC, isMusicEnabled ? 1 : 0);
            PlayerPrefs.SetFloat(PREF_MUSIC_VOL, musicVolume);
            PlayerPrefs.SetFloat(PREF_SFX_VOL, sfxVolume);
            PlayerPrefs.Save();
        }

        public void LoadFromPrefs()
        {
            participantName = PlayerPrefs.GetString(PREF_NAME, "");
            isDDAEnabled = PlayerPrefs.GetInt(PREF_DDA, 1) == 1;
            isMusicEnabled = PlayerPrefs.GetInt(PREF_MUSIC, 1) == 1;
            musicVolume = PlayerPrefs.GetFloat(PREF_MUSIC_VOL, 0.4f);
            sfxVolume = PlayerPrefs.GetFloat(PREF_SFX_VOL, 1.0f);
        }

        /// <summary>
        /// Membalik status DDA untuk eksperimen Within-Subjects.
        /// </summary>
        public void FlipDDA()
        {
            isDDAEnabled = !isDDAEnabled;
            SaveToPrefs();
            Debug.Log($"[GameSettings] DDA di-flip menjadi: {isDDAEnabled}");
        }
    }
}
