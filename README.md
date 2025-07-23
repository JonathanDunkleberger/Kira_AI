Conversational AI VTuber

![Demo of AI VTuber in action](https://github.com/DuchessGhanima/Kira_AI/blob/main/VTuber%20Demo%20-%20Kirav2.gif?raw=true)

A fully-interactive, AI-powered VTuber that engages with a live Twitch chat in real-time, featuring a dynamic personality and realistic voice synthesis.

🚀 Key Features
Real-Time Interaction: Connects directly to the Twitch API to read chat messages and respond to users live.

Generative AI Personality: Leverages the OpenAI API (GPT-4) to generate unique, context-aware, and engaging conversational responses. No two conversations are the same.

Realistic Voice Synthesis: Integrates the Microsoft Azure API to convert the AI's text responses into high-quality, natural-sounding speech.

End-to-End Pipeline: The entire process, from receiving a chat message to the AI speaking, is handled through an automated pipeline.

🛠️ Tech Stack
Core Language: Python

AI & Machine Learning: OpenAI API

Text-to-Speech (TTS): Microsoft Azure API

Real-Time Data: Twitch API

Development Environment: VS Code

🏛️ Architecture Overview
This project operates as a sequential pipeline that processes events from user interaction to AI response:

Twitch Listener: A Python script establishes a persistent connection to the Twitch API, continuously listening for new chat messages in a specified channel.

AI Core: Upon receiving a new message, the text is sent as a prompt to the OpenAI API. The API then returns a unique, generated text response.

Voice Synthesis: The generated text is passed to the Microsoft Azure API, which returns a stream of audio data representing the synthesized voice.

Audio Output: The application plays the audio stream, allowing the VTuber to "speak" its response.

⚙️ Setup & Installation
To run this project locally, you will need to have Python installed and obtain API keys for OpenAI, Microsoft Azure, and Twitch.

Clone the repository:

git clone [your-repository-url]

Install dependencies:

pip install -r requirements.txt

Configure API Keys:

Create a .env file and add your API keys.

Run the application:

python main.py
