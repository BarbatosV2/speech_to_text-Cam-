import cv2
import speech_recognition as sr
import threading
import time

# Global variable to store recognized text and timestamp
recognized_text = ""
timestamp = time.time()

def recognize_speech_from_mic():
    global recognized_text, timestamp
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                recognized_text = text
                timestamp = time.time()  # Update timestamp when new text is recognized
                print(f"Recognized Text: {text}")
            except sr.UnknownValueError:
                print("Google Web Speech API could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Web Speech API; {e}")

def show_camera():
    global recognized_text, timestamp
    cap = cv2.VideoCapture(0)

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
        frame = cv2.resize(frame, (1800, 1080 ))

        # Check if 2 seconds have passed since the text was recognized
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
