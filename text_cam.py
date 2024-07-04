import cv2
import torch
import threading
import time
import speech_recognition as sr
import torch.nn as nn
import torch.nn.functional as F

# Enhanced neural network
class EnhancedNN(nn.Module):
    def __init__(self):
        super(EnhancedNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)  # Adjust output size as needed

    def forward(self, x):
        # Convolutional layers with ReLU, max pooling, and batch normalization
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.dropout(x)
        
        # Flatten the image
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Global variables to store recognized text and timestamp
recognized_lines = []
timestamp = time.time()

def callback(recognizer, audio):
    global recognized_lines, timestamp
    try:
        text = recognizer.recognize_google(audio)
        if text:
            recognized_lines.append(text)  # Append recognized text as a new line
            if len(recognized_lines) > 2:
                recognized_lines.pop(0)  # Keep only the last two lines
            timestamp = time.time()  # Update timestamp when new text is recognized
            print(f"Recognized Text: {recognized_lines}")
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
    
    stop_listening = recognizer.listen_in_background(mic, callback)

    # Keep the main thread alive
    try:
        while True: time.sleep(0.1)
    except KeyboardInterrupt:
        stop_listening(wait_for_stop=False)

def show_camera():
    global recognized_lines, timestamp
    cap = cv2.VideoCapture(0)
    model = EnhancedNN()  # Initialize the enhanced neural network
    model.eval()  # Set the model to evaluation mode

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    cv2.namedWindow("Camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Resize the frame to a smaller size
        frame = cv2.resize(frame, (640, 480))  # Resize to 64x64 for the CNN

        # Dummy processing with PyTorch model
        # Convert frame to tensor and normalize
        tensor_frame = torch.tensor(frame, dtype=torch.float32) / 255.0
        tensor_frame = tensor_frame.permute(2, 0, 1).unsqueeze(0)  # Convert to NCHW format

        # Perform a forward pass with the enhanced neural network
        with torch.no_grad():
            output = model(tensor_frame)

        # Check if 4 seconds have passed since the text was recognized
        if time.time() - timestamp > 4:
            recognized_lines = []

        # Display the recognized text at the bottom middle of the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(recognized_lines):
            text_size = cv2.getTextSize(line, font, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = frame.shape[0] - 30 - (1 - i) * 30  # Position the text 30 pixels from the bottom, each line 30 pixels apart
            cv2.putText(frame, line, (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Camera", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create a separate thread for speech recognition
    speech_thread = threading.Thread(target=recognize_speech_from_mic)
    speech_thread.daemon = True
    speech_thread.start()

    # Start the OpenCV camera feed
    show_camera()
