import pyaudio

def find_audio_devices():
    """
    Lists all available audio output devices using pyaudio.
    """
    print("--- Finding Audio Output Devices ---")
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    
    output_devices = []
    for i in range(device_count):
        try:
            device_info = p.get_device_info_by_index(i)
            # Check if the device has output channels (is a speaker/output)
            if device_info.get('maxOutputChannels') > 0:
                output_devices.append(device_info.get('name'))
        except Exception as e:
            print(f"Could not get info for device {i}: {e}")

    if not output_devices:
        print("No audio output devices found.")
    else:
        print("Available output devices:")
        for name in output_devices:
            # Pyaudio may decode with a different encoding, so we ensure it's clean UTF-8
            clean_name = name.encode('latin-1').decode('utf-8', 'ignore')
            print(f"  - \"{clean_name}\"")
            
    print("\nCopy the exact name of your virtual cable (without the quotes)")
    print("and paste it into the VIRTUAL_AUDIO_DEVICE field in your .env file.")
    
    p.terminate()

if __name__ == "__main__":
    find_audio_devices()