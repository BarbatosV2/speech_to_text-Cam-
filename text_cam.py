import cv2
import speech_recognition as sr
import threading

# Global variable to store recognized text
recognized_text = ""

def recognize_speech_from_mic():
    global recognized_text
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
                print(f"Recognized Text: {text}")
            except sr.UnknownValueError:
                print("Google Web Speech API could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Web Speech API; {e}")

def show_camera():
    global recognized_text
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

        # Display the recognized text on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, recognized_text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

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
