import sqlite3
import numpy as np
from datetime import datetime, timedelta
import pickle
import scipy.io.wavfile as wav
import scipy.io.wavfile as wavfile
import io
import scipy.fft
import wave
import os

import createML as cml

# Constants
SAMPLE_LENGTH = 5  # seconds
FFT_POINTS = 4096
FREQUENCY_RANGE = 8000  # Hz
PARTS = 40

def process_wav_file_IO(wav_bytes_io):

    # Read the .wav file
    #sample_rate, data = wavfile.read(file_io)

    # Open the WAV file from the buffer
    with wave.open(wav_bytes_io, 'rb') as wav_file:
        # Extract audio parameters
        nchannels, sampwidth, framerate, nframes, comptype, compname = wav_file.getparams()
        sample_rate = framerate

        # Read audio data
        frames = wav_file.readframes(nframes)
        # Convert the frames to a numpy array based on the sample width and number of channels
        if sampwidth == 1:  # 8-bit audio
            dtype = np.uint8
        elif sampwidth == 2:  # 16-bit audio
            dtype = np.int16
        elif sampwidth == 3:  # 24-bit audio
            dtype = np.int32
        elif sampwidth == 4:  # 32-bit audio
            dtype = np.int32
        else:
            raise ValueError("Unsupported sample width")

        data = np.frombuffer(frames, dtype=dtype)
        if nchannels > 1:
            data = np.reshape(data, (-1, nchannels))


    print(sample_rate)
    print(len(data))
    print(data)

    # If stereo, use only one channel
    if data.ndim > 1:
        data = data[:, 0]

    # Calculate the number of samples for 5 seconds
    num_samples = SAMPLE_LENGTH * sample_rate

    # Initialize list to store moving mean values
    moving_means = []

    # Process in 5-second segments
    for start in range(0, len(data), num_samples):
        segment = data[start:start + num_samples]

        # Apply FFT
        fft_result = scipy.fft.fft(segment, n=FFT_POINTS)[:FREQUENCY_RANGE]
        fft_magnitude = np.abs(fft_result)

        # Split FFT result into 40 parts and calculate moving mean
        split_size = FFT_POINTS // PARTS
        for i in range(0, len(fft_magnitude), split_size):
            part_fft = fft_magnitude[i:i + split_size]
            moving_mean = np.mean(part_fft)
            moving_means.append(moving_mean)

    return moving_means

def create_wav_from_db(db_path, table_name):
    # Calculate the timestamp for 10 seconds ago
    time_threshold = datetime.now() - timedelta(seconds=10)
    time_threshold_str = time_threshold.strftime('%Y-%m-%d_%H-%M-%S-%f')

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # SQL query to select microphone values from the last 10 seconds
    query = f"SELECT * FROM {table_name} WHERE timestamp > '{time_threshold_str}'"

    try:
        cursor.execute(query)
        results = cursor.fetchall()

    except sqlite3.Error as e:
        print("Database error:", e)
        return None

    finally:
        conn.close()

    if results:
        # Extract only the microphone values
        mic_values = np.array([row[-1] for row in results])

        # Normalize and scale to 16-bit range
        scaled_values = np.int16(mic_values / np.max(np.abs(mic_values)) * 32767)

        # Define the sample rate
        sample_rate = 44100

        # Create a BytesIO object and write the WAV file to it
        wav_buffer = io.BytesIO()
        wav.write(wav_buffer, sample_rate, scaled_values)

        # Return the buffer containing the WAV file
        return wav_buffer.getvalue()
    else:
        return None

def create_wav_from_db_disk(db_path, table_name):

    output_folder='output_wav_files'

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Calculate the timestamp for 10 seconds ago
    time_threshold = datetime.now() - timedelta(seconds=10)
    formatted_time = time_threshold.strftime('%Y-%m-%d_%H-%M-%S-%f')

    # Query to retrieve the last 10 seconds of microphone data
    query = f"SELECT * FROM {table_name} WHERE datetime > '{formatted_time}'"
    cursor.execute(query)
    results = cursor.fetchall()

    # Close the database connection
    conn.close()

    # Process the microphone data if available
    if results:
        # Extract microphone values
        mic_values = np.array([row[-1] for row in results], dtype=np.float32)

        # Define the sample rate
        sample_rate = 44100  # Replace with the actual sample rate

        # Normalize and scale to 16-bit PCM range
        mic_values = np.int16((mic_values / np.max(np.abs(mic_values))) * 32767)

        # Generate a unique filename with folder path
        filename = os.path.join(output_folder, datetime.now().strftime("output_%Y-%m-%d_%H-%M-%S.wav"))

        # Write the data to the .wav file
        wav.write(filename, sample_rate, mic_values)
        print(f"WAV file created: {filename}")

        return filename
    else:
        print("No recent data available.")
        return None



def analyse_database():

    db_path = 'local_data.db'  # Replace with your database path
    table_name = 'sensor_data'  # Replace with your table name
    wav_file = create_wav_from_db_disk(db_path, table_name)

    # Create an in-memory file-like object from wav_bytes
    #wav_bytes_io = io.BytesIO(wav_bytes)

    test_audio_movmean = cml.process_wav_file(wav_file)

    # Load the trained model
    with open('random_forest_model.pkl', 'rb') as file:
        clf = pickle.load(file)

    # Predict using the loaded model
    predictions = clf.predict(test_audio_movmean)

    return predictions

