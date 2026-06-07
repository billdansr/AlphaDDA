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
        public bool isP2Human = false;

        private const string PREF_NAME = "Name";
        private const string PREF_DDA = "DDA";
        private const string PREF_MUSIC = "Music";

        public void SaveToPrefs()
        {
            PlayerPrefs.SetString(PREF_NAME, participantName);
            PlayerPrefs.SetInt(PREF_DDA, isDDAEnabled ? 1 : 0);
            PlayerPrefs.SetInt(PREF_MUSIC, isMusicEnabled ? 1 : 0);
            PlayerPrefs.Save();
        }

        public void LoadFromPrefs()
        {
            participantName = PlayerPrefs.GetString(PREF_NAME, "");
            isDDAEnabled = PlayerPrefs.GetInt(PREF_DDA, 1) == 1;
            isMusicEnabled = PlayerPrefs.GetInt(PREF_MUSIC, 1) == 1;
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
