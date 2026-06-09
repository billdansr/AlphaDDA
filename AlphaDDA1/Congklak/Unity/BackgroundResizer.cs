using UnityEngine;

namespace CongklakAI
{
    /// <summary>
    /// Komponen untuk menyesuaikan skala SpriteRenderer agar menutupi seluruh layar.
    /// Cocok digunakan untuk background pada kamera Orthographic.
    /// </summary>
    [RequireComponent(typeof(SpriteRenderer))]
    public class BackgroundResizer : MonoBehaviour
    {
        private SpriteRenderer spriteRenderer;

        void Start()
        {
            spriteRenderer = GetComponent<SpriteRenderer>();
            Resize();
        }

        /// <summary>
        /// Menghitung ulang skala objek berdasarkan dimensi kamera dan sprite.
        /// </summary>
        public void Resize()
        {
            if (spriteRenderer == null || spriteRenderer.sprite == null) return;

            Camera cam = Camera.main;
            if (cam == null)
            {
                Debug.LogWarning("[BackgroundResizer] Main Camera tidak ditemukan.");
                return;
            }

            // 1. Reset scale agar perhitungan bounds murni dari ukuran asli sprite
            transform.localScale = Vector3.one;

            // 2. Dapatkan ukuran sprite dalam unit World
            float spriteWidth = spriteRenderer.sprite.bounds.size.x;
            float spriteHeight = spriteRenderer.sprite.bounds.size.y;

            // 3. Hitung tinggi dan lebar layar dalam unit World
            // OrthographicSize adalah setengah dari tinggi layar
            float worldScreenHeight = cam.orthographicSize * 2.0f;
            float worldScreenWidth = worldScreenHeight * cam.aspect;

            // 4. Terapkan skala baru (Stretch to Fill)
            transform.localScale = new Vector3(
                worldScreenWidth / spriteWidth,
                worldScreenHeight / spriteHeight,
                1.0f
            );
        }
    }
}