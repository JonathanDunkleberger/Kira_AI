# 🎤 Kira AI VTuber (Triad Architecture)

![Demo of AI VTuber in action](https://github.com/JonathanDunkleberger/Kira_AI/blob/main/VTuber%20Demo%20-%20Kirav3.gif?raw=true)

**Kira** is a highly advanced, self-aware AI VTuber companion running on local hardware (RTX 50-series optimized). She features a unique "Triad" architecture that separates her relationship with her Creator (Voice), her Audience (Chat), and her own internal Identity.

---

## ✨ Key Features

🧠 **Cognitive Memory System**  
- **Fact Extraction**: Automatically learns facts ("Jonny likes Evangelion") from voice conversations using structured JSON extraction.
- **Project Tracking**: Remembers ongoing tasks and projects over time.
- **Dual Database**: Separates raw conversation logs (`turns`) from distilled knowledge (`facts`).

🎭 **"Triad" Personality Architecture**  
- **The Creator (Voice)**: High-trust channel. Kira listens to you and learns from you.
- **The Mob (Twitch)**: High-chaos channel. Kira entertains and roasts chat, but does *not* learn long-term facts from them (safety).
- **The AI (Identity)**: Defined strictly in `personality.txt`. She is sassy, self-aware, and refuses to be a robotic assistant.

⚡ **High-Performance Local Inference**  
- **LLM**: Llama-3 (8B) running locally via `llama-cpp-python` with Flash Attention.
- **STT**: Faster-Whisper (medium.en) for real-time transcription.
- **TTS**: Azure Neural TTS (standard) or ElevenLabs (premium).

---

## 🚀 Setup Guide

### 1. Requirements
- Python 3.10+
- NVIDIA GPU (RTX 3060 or better recommended) with CUDA drivers.
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (C++ CMake tools) for compiling llama-cpp.

### 2. Installation
```bash
git clone https://github.com/JonathanDunkleberger/Kira_AI.git
cd Kira_AI
pip install -r requirements.txt
```
*Note: You may need to reinstall `llama-cpp-python` with specific CUDA flags if GPU acceleration fails.*

### 3. Configuration
1. Copy `.env.example` to `.env`.
2. Fill in your keys (Azure, Twitch, etc.).
3. Download a GGUF model (e.g., `Meta-Llama-3-8B-Instruct-Q5_K_M.gguf`) to the `models/` folder.
4. Update `LLM_MODEL_PATH` in `.env` to match your filename.

### 4. Customization
- **Personality**: Edit `personality.txt`. This is the single source of truth for who Kira is.
- **Rules**: Edit `prompt_rules.py` for output formatting (no brackets, no emojis, etc.).

### 5. Run
```bash
python bot.py
```

---

## 🛠️ Architecture Notes

- **`bot.py`**: The orchestrator. Handles the event loop, audio capture, and message routing.
- **`memory_extractor.py`**: The "Hippocampus". Analyzing voice input to form long-term memories.
- **`memory.py`**: Interface for ChromaDB (Vector Database).
- **`ai_core.py`**: The "Cortex". Handles LLM inference, TTS generation, and STT.

---

## 🔒 Privacy & Safety

- **Local First**: All inference happens on your machine.
- **Memory Safety**: Twitch chat typically cannot poison Kira's long-term memory due to the "Voice-Only" extraction gate.
- **Secrets**: API keys are stored in `.env` and excluded from git.
