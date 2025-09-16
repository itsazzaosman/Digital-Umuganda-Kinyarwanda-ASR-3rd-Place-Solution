# Kinyarwanda ASR: 3rd Place Solution 🏆

This repository contains the code and methodology for the 3rd place winning solution in the **Digital Umuganda Kinyarwanda ASR Challenge** [Kaggle competition](https://www.kaggle.com/competitions/kinyarwanda-automatic-speech-recognition-track-b). The primary goal was to develop a high-accuracy Automatic Speech Recognition (ASR) model for the Kinyarwanda language.

---

## 🚀 Methodology and Experiments

To achieve the best possible performance, we explored several state-of-the-art ASR architectures. Our approach was highly experimental, involving training and fine-tuning multiple models to compare their effectiveness on the Kinyarwanda dataset.

### Models Explored
- **Whisper**: We fine-tuned various sizes of OpenAI's Whisper model, which provides a strong baseline for many languages (using the default tokenizer and a customized tokenizer)or Kinyarwanda).
- **Conformer**: We experimented with Conformer-based architectures, known for their excellent ability to capture both local and global features in audio.
- **Parakeet (Winning Model)**: After extensive evaluation, the **NVIDIA Parakeet (CTM)** model family delivered the CombinedError = 0.4 × WER + 0.6 × CER
Score = (1 – CombinedError) × 100, and became the foundation of our final submission. The code and notebooks related to this model can be found in the `model_Parakeet/` directory.

### Language Model Integration (Attempted)
We also attempted to further improve the model's accuracy by incorporating a language model (LM) for post-processing.

- **KenLM & Beam Search**: We trained a 5-gram KenLM model on a custom Kinyarwanda text corpus. The goal was to use this LM with a beam search decoder to refine the model's raw transcriptions and correct common grammatical errors.
- **Challenges**: Unfortunately, we encountered significant technical challenges and persistent errors during the integration phase, particularly with the decoding libraries (`pyctcdecode`). Due to these unresolved issues, we were unable to successfully incorporate the KenLM beam search decoder into our final pipeline. This remains a promising area for future improvement.

---

## 🛠️ Setup and Installation

To reproduce our environment and run the code, please follow these steps.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/itsazzaosman/DIgital-Umuganda-Kinyarwanda-ASR-3rd-Place-Solution.git](https://github.com/itsazzaosman/DIgital-Umuganda-Kinyarwanda-ASR-3rd-Place-Solution.git)
    cd DIgital-Umuganda-Kinyarwanda-ASR-3rd-Place-Solution
    ```

2.  **Create the Conda environment:**
    The required environment files are located in the `envs/` directory. Use the environment file corresponding to the Parakeet model to replicate our winning setup.
    ```bash
    # Example using the Parakeet environment file
    conda env create -f envs/parakeet_env.yml 
    conda activate parakeet_env
    ```

---



## 💡 Future Work
-   **Resolve KenLM Integration**: The most immediate next step would be to debug the beam search decoder integration. Successfully applying the trained KenLM could significantly reduce the WER.
-   **Model Ensembling**: Combining the predictions from the top-performing models (e.g., Parakeet and Whisper) could yield further performance gains.







