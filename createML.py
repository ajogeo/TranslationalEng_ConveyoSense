import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import scipy.io.wavfile as wavfile
import scipy.fft

# Constants
SAMPLE_LENGTH = 5  # seconds
FFT_POINTS = 4096
FREQUENCY_RANGE = 8000  # Hz
PARTS = 40

def process_wav_file(file_path):

    # Read the .wav file
    sample_rate, data = wavfile.read(file_path)

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

def extract_input_movavg():

    mm_set = np.empty((0,123))
    label_set = np.empty((0))

    file_path_1 = 'datafiles\Rolo_Bom_1.wav' 
    moving_means_1 = process_wav_file(file_path_1)
    mm_set = np.vstack((mm_set,  moving_means_1))
    label_set = np.hstack((label_set, 1))

    file_path_2 = 'datafiles\Rolo_Bom_2.wav' 
    moving_means_2 = process_wav_file(file_path_2)
    mm_set = np.vstack((mm_set,  moving_means_2))
    label_set = np.hstack((label_set, 1))

    file_path_3 = 'datafiles\Rolo_Bom_3.wav' 
    moving_means_3 = process_wav_file(file_path_3)
    mm_set = np.vstack((mm_set,  moving_means_3))
    label_set = np.hstack((label_set, 1))

    file_path_4 = 'datafiles\Rolo_Bom_4.wav' 
    moving_means_4 = process_wav_file(file_path_4)
    mm_set = np.vstack((mm_set,  moving_means_4))
    label_set = np.hstack((label_set, 1))

    file_path_5 = 'datafiles\Rolo_Bom_5.wav' 
    moving_means_5 = process_wav_file(file_path_5)
    mm_set = np.vstack((mm_set,  moving_means_5))
    label_set = np.hstack((label_set, 1))

    file_path_6 = 'datafiles\Rolo_Bom_6.wav' 
    moving_means_6 = process_wav_file(file_path_6)
    mm_set = np.vstack((mm_set,  moving_means_6))
    label_set = np.hstack((label_set, 1))

    file_path_7 = 'datafiles\Rolo_Bom_7.wav' 
    moving_means_7 = process_wav_file(file_path_7)
    mm_set = np.vstack((mm_set,  moving_means_7))
    label_set = np.hstack((label_set, 1))

    file_path_8 = 'datafiles\Rolo_Bom_8.wav' 
    moving_means_8 = process_wav_file(file_path_8)
    mm_set = np.vstack((mm_set,  moving_means_8))
    label_set = np.hstack((label_set, 1))

    file_path_9 = 'datafiles\Rolo_Bom_9.wav' 
    moving_means_9 = process_wav_file(file_path_9)
    mm_set = np.vstack((mm_set,  moving_means_9))
    label_set = np.hstack((label_set, 1))

    file_path_10 = 'datafiles\Rolo_Bom_10.wav' 
    moving_means_10 = process_wav_file(file_path_10)
    mm_set = np.vstack((mm_set,  moving_means_10))
    label_set = np.hstack((label_set, 1))

    ########################################################

    file_path_d1 = 'datafiles\Rolo_Ruim_Def-1.wav'
    moving_means_d1 = process_wav_file(file_path_d1)
    mm_set = np.vstack((mm_set,  moving_means_d1))
    label_set = np.hstack((label_set, 0))

    file_path_d2 = 'datafiles\Rolo_Ruim_Def-2.wav'
    moving_means_d2 = process_wav_file(file_path_d2)
    mm_set = np.vstack((mm_set,  moving_means_d2))
    label_set = np.hstack((label_set, 0))

    file_path_d3 = 'datafiles\Rolo_Ruim_Def-3.wav'
    moving_means_d3 = process_wav_file(file_path_d3)
    mm_set = np.vstack((mm_set,  moving_means_d3))
    label_set = np.hstack((label_set, 0))

    file_path_d4 = 'datafiles\Rolo_Ruim_Def-4.wav'
    moving_means_d4 = process_wav_file(file_path_d4)
    mm_set = np.vstack((mm_set,  moving_means_d4))
    label_set = np.hstack((label_set, 0))

    file_path_d5 = 'datafiles\Rolo_Ruim_Def-5.wav'
    moving_means_d5 = process_wav_file(file_path_d5)
    mm_set = np.vstack((mm_set,  moving_means_d5))
    label_set = np.hstack((label_set, 0))

    file_path_d6 = 'datafiles\Rolo_Ruim_Def-6.wav'
    moving_means_d6 = process_wav_file(file_path_d6)
    mm_set = np.vstack((mm_set,  moving_means_d6))
    label_set = np.hstack((label_set, 0))

    file_path_d7 = 'datafiles\Rolo_Ruim_Def-7.wav'
    moving_means_d7 = process_wav_file(file_path_d7)
    mm_set = np.vstack((mm_set,  moving_means_d7))
    label_set = np.hstack((label_set, 0))

    file_path_d8 = 'datafiles\Rolo_Ruim_Def-8.wav'
    moving_means_d8 = process_wav_file(file_path_d8)
    mm_set = np.vstack((mm_set,  moving_means_d8))
    label_set = np.hstack((label_set, 0))

    file_path_d9 = 'datafiles\Rolo_Ruim_Def-9.wav'
    moving_means_d9 = process_wav_file(file_path_d9)
    mm_set = np.vstack((mm_set,  moving_means_d9))
    label_set = np.hstack((label_set, 0))
    
    file_path_d10 = 'datafiles\Rolo_Ruim_Def-10.wav'
    moving_means_d10 = process_wav_file(file_path_d10)
    mm_set = np.vstack((mm_set,  moving_means_d10))
    label_set = np.hstack((label_set, 0))

    return mm_set, label_set

def train_RF_MLmodel(mm_set, label_set):

    idx = np.random.permutation(20)

    mm_set = mm_set[idx,:]
    label_set = label_set[idx]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = mm_set, mm_set, label_set, label_set

    # Initialize and train the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the classifier
    #print("Classification Report:")
    #print(classification_report(y_test, y_pred))
    #print("Accuracy:", accuracy_score(y_test, y_pred))

    with open('random_forest_model.pkl', 'wb') as file:
        pickle.dump(clf, file)

def main():

    mm_set, label_set = extract_input_movavg()

    train_RF_MLmodel(mm_set, label_set)

    """
    # Combine data and labels
    X = np.vstack((moving_means_NDI, moving_means_DI))
    y = np.hstack((labels_NDI, labels_DI))

    # Initialize the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    clf.fit(X, y)

    # Save the trained model
    with open('random_forest_model.pkl', 'wb') as file:
        pickle.dump(clf, file)

    """

if __name__ == "__main__":
    main()
