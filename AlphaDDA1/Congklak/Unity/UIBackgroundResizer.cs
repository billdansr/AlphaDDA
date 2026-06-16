using UnityEngine;

namespace CongklakAI
{
    /// <summary>
    /// Menyesuaikan RectTransform agar selalu memenuhi area Canvas (Stretch to Fill).
    /// Digunakan untuk UI Image yang bertindak sebagai background.
    /// </summary>
    [RequireComponent(typeof(RectTransform))]
    public class UIBackgroundResizer : MonoBehaviour
    {
        void Awake()
        {
            ApplyFullStretch();
        }

        [ContextMenu("Apply Full Stretch")]
        public void ApplyFullStretch()
        {
            RectTransform rect = GetComponent<RectTransform>();
            
            // Mengatur jangkar (Anchors) ke mode stretch (0 hingga 1)
            rect.anchorMin = Vector2.zero;
            rect.anchorMax = Vector2.one;
            
            // Mengatur jarak (Offsets) ke nol agar menempel tepat di pinggir Canvas
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
        }
    }
}