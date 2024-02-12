
import cv2
import mediapipe as mp

def personSkeleton(main, rest):
    mpDrawing.draw_landmarks(main, rest, mpPose.POSE_CONNECTIONS)

if __name__ == "__main__":
    video_path = "./inputVideo.mp4"
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    mpDrawing = mp.solutions.drawing_utils

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outputPath = "./outputVideo.mp4"
    finalVideo = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            personSkeleton(frame, results.pose_landmarks)

        finalVideo.write(frame)

    video.release()
    finalVideo.release()
    cv2.destroyAllWindows()

    print(f"Processing completed")
