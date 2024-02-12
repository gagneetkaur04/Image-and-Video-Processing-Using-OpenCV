import cv2
import numpy as np

def redDotsInGreenCircle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    return frame

if __name__ == "__main__":
    video_path = "./inputVideo.mp4"
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outputPath = "./outputVideo.mp4"
    finalVideo = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        result_frame = redDotsInGreenCircle(frame)
        finalVideo.write(result_frame)

    video.release()
    finalVideo.release()
    cv2.destroyAllWindows()

    print(f"Processing completed.")
