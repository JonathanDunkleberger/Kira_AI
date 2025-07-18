# test_tts.py – Verbose pyttsx3 TTS test

import pyttsx3

def main():
    # explicitly use the Windows SAPI5 driver
    engine = pyttsx3.init('sapi5')

    # list available voices
    voices = engine.getProperty('voices')
    print("Available voices:")
    for i, v in enumerate(voices):
        print(f"  [{i}] {v.name} — {v.id}")

    # select the first voice (you can change the index if needed)
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 150)        # words per minute
    engine.setProperty('volume', 1.0)      # volume: 0.0 to 1.0

    print("→ Speaking now...")
    engine.say("Testing one two three. Can you hear me now?")
    engine.runAndWait()
    print("→ Done speaking.")

if __name__ == "__main__":
    main()
