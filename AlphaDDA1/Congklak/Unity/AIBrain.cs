using UnityEngine;

using System.Linq;
using Unity.InferenceEngine;

namespace CongklakAI
{
    public class AIBrain : MonoBehaviour
    {
        [Header("Model Settings")]
        [Tooltip("The exported CongklakAlphaDDA.onnx file")]
        public Unity.InferenceEngine.ModelAsset modelAsset;
        
        private Worker worker;
        private Model runtimeModel;

        void Awake()
        {
            if (modelAsset != null)
            {
                // Load the model
                runtimeModel = ModelLoader.Load(modelAsset);
                
                // Create a worker. GPU is preferred for ResNets
                // Use BackendType.GPUCompute for modern Android/Unity 6
                // Fallback to CPU if needed
                worker = new Worker(runtimeModel, BackendType.GPUCompute);
            }
        }

        /// <summary>
        /// Performs inference on the given canonical state and returns (Policy, Value).
        /// </summary>
        public (float[] pi, float v) Predict(float[,,] state)
        {
            // Input shape is [1, 3, 2, 8]
            TensorShape shape = new(1, 3, 2, 8);
            
            // Flatten 3D array to 1D for Tensor creation
            float[] flatten = new float[1 * 3 * 2 * 8];
            int idx = 0;
            for (int c = 0; c < 3; c++)
                for (int x = 0; x < 2; x++)
                    for (int y = 0; y < 8; y++)
                        flatten[idx++] = state[c, x, y];

            using Tensor<float> inputTensor = new(shape, flatten);
            
            // Execute model
            // Input name MUST match the 'input_names' in Python export: 'input_board'
            worker.Schedule(inputTensor);

            // Get outputs
            // Names MUST match the 'output_names' in Python export
            using Tensor<float> piTensor = worker.PeekOutput("output_pi") as Tensor<float>;
            using Tensor<float> vTensor = worker.PeekOutput("output_v") as Tensor<float>;

            // Policy is log_softmax, so we convert to probabilities: exp(pi)
            float[] logPi = piTensor.DownloadToArray();
            float[] pi = logPi.Select(x => Mathf.Exp(x)).ToArray();
            
            float v = vTensor.DownloadToArray()[0];

            return (pi, v);
        }

        void OnDestroy()
        {
            worker?.Dispose();
        }
    }
}
