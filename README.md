# 🧠 Kira

![Demo of AI VTuber in action](https://github.com/JonathanDunkleberger/Kira_AI/blob/main/VTuber%20Demo%20-%20Kirav3.gif?raw=true)

> **A sophisticated, multimodal cognitive agent designed for real-time interaction, autonomous decision making, and content creation.**

**Kira** is a statement implementation of a modern local-first AI agent. Unlike standard chatbots that simply wait for a prompt and respond, Kira possesses **agency**, **long-term memory**, and **multimodal sensory awareness**. She listens to voice, watches your screen, remembers facts about you, browses the web, and integrates directly with games and streaming platforms.

Designed to run primarily on local hardware (RTX-optimized), she bridges the gap between Large Language Models (LLMs) and true digital companionship.

---

## ✨ Core Capabilities

### 1. ⚡ Local-First Cognitive Engine
- **Inference**: Powered by **Llama 3 / 3.1 (8B)** running locally via `llama-cpp-python`.
- **Latency**: Optimization with Flash Attention for real-time conversational speeds.
- **Privacy**: Your conversations and data stay on your machine.

### 2. 🧠 Long-Term Memory (Hippocampus)
- **Vector Database**: Uses **ChromaDB** to store and retrieve memories semantically.
- **Fact Extraction**: Automatically analyzes voice conversations to extract permanent facts (e.g., "User is working on a Python project") rather than just storing raw text.
- **Contextual Recall**: Dynamically pulls relevant past memories into the context window based on the current conversation.

### 3. 👁️ Multimodal Sensory System
- **Vision Agent**: Uses Vision-connected LLMs to "see" the user's screen, understanding context from games, code editors, or videos.
- **Hearing**: State-of-the-art **Faster-Whisper** implementation for accurate, real-time voice transcription.
- **Speech**: High-fidelity TTS (Text-to-Speech) via **Azure Neural** or **ElevenLabs** for emotive vocal expression.

### 4. 🎮 Gaming & Streaming Integration
- **Universal Media Bridge**: Monitors game logs (Minecraft, Hytale, etc.) to react to in-game events in real-time.
- **Twitch Integration**: Full chat interaction, poll management, and moderation capabilities.
- **Game Mode Controller**: Automatically detects active game windows to switch context and behavior.

### 5. 🎵 Media & Browsing
- **DJ Mode**: Integrated `music_tools` allow the agent to search YouTube and play music directly via `mpv` based on natural language requests.
- **Web Research**: Autonomous **Google Custom Search** capabilities to look up real-time information when it doesn't know an answer.

### 6. ⚛️ Proactive Agency
- **Thought Loop**: The agent doesn't just wait for input. It runs a background "thoughts" loop, allowing it to speak up, ask questions, or comment on recognized visual events without being prompted.

---

## 🏗️ Technical Architecture

This project moves beyond simple "User Input -> LLM -> Output" scripts into a modular agentic architecture:

| Component | Responsibility | Tech Stack |
|-----------|----------------|------------|
| **Cortex** (`ai_core.py`) | Orchestrates the neurological loop. Manages LLM inference, context windowing, and tool decision making. | `llama-cpp-python`, `torch` |
| **Hippocampus** (`memory.py`) | Semantic retrieval and consolidation. Separates episodic memory (interactions) from semantic memory (facts). | `ChromaDB`, `sqlite3` |
| **Vision** (`vision_agent.py`) | Snapshots screen context, processes via VLM, and injects descriptions into the cognitive stream. | `Pillow`, `OpenAI Vision API` |
| **Senses** (`bot.py`) | Real-time VAD (Voice Activity Detection), STT pipeline, and audio output mixing. | `Faster-Whisper`, `PyAudio`, `webrtcvad` |
| **Dashboard** (`dashboard.py`) | Professional GUI for monitoring internal state, vision feed, and manual overrides. | `CustomTkinter` |

---

## 🚀 Setup & Installation

### Prerequisites
- **Python**: 3.10 or higher.
- **GPU**: NVIDIA RTX 3060 or better (recommended for local inference).
- **External Tools**: 
  - `mpv` (for music playback).
  - [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (required for compiling llama-cpp).

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/JonathanDunkleberger/Kira_AI.git
   cd Kira_AI
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you have a GPU, ensure you install the CUDA-enabled version of `llama-cpp-python`.*

3. **Model Setup**
   - Download a GGUF model (e.g., `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`) from HuggingFace.
   - Place it in the `models/` directory.

4. **Configuration**
   - Create a `.env` file (see `.env.example`).
   - Add your API keys (ElevenLabs, Azure, OpenAI for vision, Google Search).
   - Update `config.py` if necessary for pathing.

5. **Launch**
   ```bash
   python dashboard.py
   ```
   *Running `dashboard.py` will launch the GUI and the bot process together.*

---

## 🛠️ Customization

- **Personality**: The agent's core identity is defined in `personality.txt`. This is a natural language prompt that defines her name, backstory, and behavioral traits.
- **Persona Styling**: Modify `persona.py` to adjust emotional weights and reaction thresholds.
- **Rules**: `prompt_rules.py` enforces formatting constraints (e.g., "Do not use emojis", "Keep responses under 2 sentences").

---

## 🔮 Future Roadmap

- **GraphRAG**: Implementing Graph-based Retrieval Augmented Generation for better relationship tracking between memories.
- **Local Vision Model**: Replacing the API-based vision agent with a local quantized LLaVA or similar model for full offline capability.
- **Live2D / VTube Studio Model**: Direct WebSocket integration to drive a Live2D avatar's mouth and expressions based on emotional state.
