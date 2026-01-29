import cv2
import os

def save_frames_as_images(camera_index=0, output_folder='frames', image_format='jpg'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open a connection to the specified camera
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        return

    frame_number = 0  # Counter for image filenames

    # Main loop to capture frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame was not grabbed, break the loop
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Display the captured frame
        cv2.imshow('Camera Stream', frame)

        if (cv2.waitKey(1) & 0xFF == ord('w')):
            # Define the filename for the current frame
            filename = os.path.join(output_folder, f'frame_{frame_number:04d}.{image_format}')

            # Save the current frame as an image file
            cv2.imwrite(filename, frame)

            # Increment the frame counter
            frame_number += 1

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Specify the output folder and image format
    save_frames_as_images(output_folder='frames', image_format='jpg')