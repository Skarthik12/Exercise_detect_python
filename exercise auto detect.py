import cv2
import mediapipe as mp # type: ignore
import numpy as np
from flask import Flask, jsonify, render_template, Response, request # type: ignore
from collections import deque

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize counters, stages, and buffers for all exercises
exercise_counts = {
    "pushup": 0,
    "hammer_curl": 0,
    "bench_press": 0,
    "lateral_raise": 0,
    "bent_over_row": 0,
    "lunge": 0,
    "side_plank": 0,
    "reverse_swing": 0,
    "barbell_squat": 0,
    "situp": 0
}
exercise_stages = {exercise: None for exercise in exercise_counts.keys()}
angle_buffers = {exercise: deque(maxlen=5) for exercise in exercise_counts.keys()}  # Smoothing buffer for each exercise angle

# Variable to store the currently selected exercise
current_exercise = None

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def smooth_angle(angle, exercise):
    """Smooths the angle using a moving average for the specified exercise."""
    angle_buffers[exercise].append(angle)
    return sum(angle_buffers[exercise]) / len(angle_buffers[exercise])

def detect_exercise(landmarks):
    global exercise_counts, exercise_stages, current_exercise

    if current_exercise == "pushup":
        shoulder_angle = smooth_angle(calculate_angle(
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        ), "pushup")
        hip_angle = smooth_angle(calculate_angle(
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ), "pushup")
        if shoulder_angle > 160 and hip_angle > 160:
            exercise_stages["pushup"] = "up"
        if shoulder_angle < 90 and hip_angle < 130 and exercise_stages["pushup"] == "up":
            exercise_stages["pushup"] = "down"
            exercise_counts["pushup"] += 1

    elif current_exercise == "hammer_curl":
        curl_elbow_angle = smooth_angle(calculate_angle(
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        ), "hammer_curl")
        if curl_elbow_angle > 160:
            exercise_stages["hammer_curl"] = "down"
        if curl_elbow_angle < 30 and exercise_stages["hammer_curl"] == "down":
            exercise_stages["hammer_curl"] = "up"
            exercise_counts["hammer_curl"] += 1

    elif current_exercise == "bench_press":
        shoulder_angle = smooth_angle(calculate_angle(
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        ), "bench_press")
        elbow_angle = smooth_angle(calculate_angle(
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        ), "bench_press")
        if shoulder_angle > 160 and elbow_angle > 160:
            exercise_stages["bench_press"] = "down"
        if shoulder_angle < 90 and elbow_angle < 90 and exercise_stages["bench_press"] == "down":
            exercise_stages["bench_press"] = "up"
            exercise_counts["bench_press"] += 1

    elif current_exercise == "lateral_raise":
        lateral_raise_angle = smooth_angle(calculate_angle(
            [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        ), "lateral_raise")
        if lateral_raise_angle > 160:
            exercise_stages["lateral_raise"] = "down"
        if lateral_raise_angle < 90 and exercise_stages["lateral_raise"] == "down":
            exercise_stages["lateral_raise"] = "up"
            exercise_counts["lateral_raise"] += 1

    elif current_exercise == "bent_over_row":
        left_arm_angle = smooth_angle(calculate_angle(
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        ), "bent_over_row")
        if left_arm_angle > 160:
            exercise_stages["bent_over_row"] = "down"
        if left_arm_angle < 90 and exercise_stages["bent_over_row"] == "down":
            exercise_stages["bent_over_row"] = "up"
            exercise_counts["bent_over_row"] += 1

    elif current_exercise == "lunge":
        left_knee_angle = smooth_angle(calculate_angle(
            [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        ), "lunge")
        if left_knee_angle > 160:
            exercise_stages["lunge"] = "up"
        if left_knee_angle < 90 and exercise_stages["lunge"] == "up":
            exercise_stages["lunge"] = "down"
            exercise_counts["lunge"] += 1

    elif current_exercise == "side_plank":
        hip_angle = smooth_angle(calculate_angle(
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        ), "side_plank")
        if 150 < hip_angle < 180:
            exercise_stages["side_plank"] = "aligned"
            exercise_counts["side_plank"] += 1

    elif current_exercise == "reverse_swing":
        arm_swing_angle = smooth_angle(calculate_angle(
            [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        ), "reverse_swing")
        if arm_swing_angle > 160:
            exercise_stages["reverse_swing"] = "down"
        if arm_swing_angle < 60 and exercise_stages["reverse_swing"] == "down":
            exercise_stages["reverse_swing"] = "up"
            exercise_counts["reverse_swing"] += 1

    elif current_exercise == "barbell_squat":
        squat_angle = smooth_angle(calculate_angle(
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ), "barbell_squat")
        if squat_angle > 160:
            exercise_stages["barbell_squat"] = "up"
        if squat_angle < 90 and exercise_stages["barbell_squat"] == "up":
            exercise_stages["barbell_squat"] = "down"
            exercise_counts["barbell_squat"] += 1

    elif current_exercise == "situp":
        hip_angle = smooth_angle(calculate_angle(
            [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        ), "situp")
        if hip_angle > 160:
            exercise_stages["situp"] = "down"
        if hip_angle < 90 and exercise_stages["situp"] == "down":
            exercise_stages["situp"] = "up"
            exercise_counts["situp"] += 1

    return current_exercise.replace("_", " ").title(), exercise_counts[current_exercise] if current_exercise else (None, 0)


def generate_frames():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                if current_exercise:
                    exercise_type, _ = detect_exercise(landmarks)
                    if exercise_type:
                        # Color conversion from ARGB (191, 40, 112, 152) to BGR
                        color_bgr = (152, 112, 40)  # OpenCV uses BGR order
                        cv2.putText(image, f"{exercise_type}", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Error: {e}")

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/set_exercise', methods=['POST'])
def set_exercise():
    global current_exercise
    data = request.get_json()
    selected_exercise = data.get('exercise')

    if selected_exercise in exercise_counts:
        current_exercise = selected_exercise
        # Reset the count for the selected exercise
        exercise_counts[current_exercise] = 0
        return jsonify({"status": "success", "current_exercise": current_exercise}), 200
    else:
        return jsonify({"status": "error", "message": "Invalid exercise type"}), 400

@app.route('/get_exercise_count', methods=['GET'])
def get_exercise_count():
    if current_exercise:
        return jsonify({current_exercise: exercise_counts[current_exercise]})
    else:
        return jsonify({"error": "No exercise selected"}), 400

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
