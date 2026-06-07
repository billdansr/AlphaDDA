using UnityEngine;

namespace CongklakAI
{
    public class GlobalMusicPlayer : MonoBehaviour
    {
        public GameSettings settings;
        private AudioSource audioSource;
        private static GlobalMusicPlayer instance;

        void Awake()
        {
            // Singleton Pattern: Memastikan hanya ada satu musik player di seluruh game
            if (instance != null)
            {
                Destroy(gameObject);
                return;
            }
            instance = this;
            DontDestroyOnLoad(gameObject);

            audioSource = gameObject.AddComponent<AudioSource>();
            if (settings == null) settings = Resources.Load<GameSettings>("GameSettings");
        }

        void Start()
        {
            UpdateMusicState();
        }

        public void UpdateMusicState()
        {
            if (settings == null || settings.backgroundMusic == null) return;

            if (settings.isMusicEnabled)
            {
                audioSource.volume = settings.musicVolume;

                if (audioSource.clip != settings.backgroundMusic)
                {
                    audioSource.clip = settings.backgroundMusic;
                    audioSource.loop = true;
                }
                
                if (!audioSource.isPlaying)
                    audioSource.Play();
            }
            else
            {
                audioSource.Stop();
            }
        }

        // Memudahkan pemanggilan dari mana saja
        public static void Refresh()
        {
            if (instance != null) instance.UpdateMusicState();
        }
    }
}