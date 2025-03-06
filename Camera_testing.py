import cv2

cap = cv2.VideoCapture(1)  # Change index if needed
image_count = 0  # Counter for captured images

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Space bar to capture images
        image_count += 1
        filename = f"cal/image{image_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

        if image_count == 2:  # Stop after capturing 2 images
            break

    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
