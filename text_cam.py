import cv2
import torch
import threading
import time
import speech_recognition as sr
import torch.nn as nn

# Dummy neural network for demonstration
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(640 * 480 * 3, 10)  # Assuming input is a flattened image

    def forward(self, x):
        x = x.view(-1, 640 * 480 * 3)  # Flatten the image
        x = self.fc(x)
        return x

# Global variable to store recognized text and timestamp
recognized_text = ""
timestamp = time.time()

def callback(recognizer, audio):
    global recognized_text, timestamp
    try:
        text = recognizer.recognize_google(audio)
        recognized_text += " " + text  # Append recognized text
        timestamp = time.time()  # Update timestamp when new text is recognized
        print(f"Recognized Text: {recognized_text}")
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
    global recognized_text, timestamp
    cap = cv2.VideoCapture(0)
    model = SimpleNN()  # Initialize the neural network
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
        frame = cv2.resize(frame, (640, 480))

        # Dummy processing with PyTorch model
        # Convert frame to tensor and normalize
        tensor_frame = torch.tensor(frame, dtype=torch.float32) / 255.0
        tensor_frame = tensor_frame.permute(2, 0, 1).unsqueeze(0)  # Convert to NCHW format

        # Perform a dummy forward pass
        with torch.no_grad():
            output = model(tensor_frame.reshape(1, -1))

        # Check if 4 seconds have passed since the text was recognized
        if time.time() - timestamp > 4:
            recognized_text = ""

        # Display the recognized text at the bottom middle of the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(recognized_text, font, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 30  # Position the text 30 pixels from the bottom
        cv2.putText(frame, recognized_text, (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

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
