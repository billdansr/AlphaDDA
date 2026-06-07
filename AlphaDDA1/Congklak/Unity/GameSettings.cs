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

        [Header("Global Audio Assets")]
        public AudioClip backgroundMusic;

        [Header("Global Audio Assets")]
        public AudioClip buttonClickSound;

        private const string PREF_NAME = "Name";
        private const string PREF_DDA = "DDA";
        private const string PREF_MUSIC = "Music";
        private const string PREF_MUSIC_VOL = "MusicVolume";
        private const string PREF_SFX_VOL = "SFXVolume";

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
