import cv2

def stream_webcam():
    # Open a connection to the webcam (commonly the device ID 0)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Capture frames continuously from the webcam
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is read correctly
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
 
        # Display the captured frame
        cv2.imshow('Webcam Stream', frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream_webcam()