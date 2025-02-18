from flask import Flask, jsonify, request
from collections import deque

app = Flask(__name__)

# Initialize counters, stages, and buffers for all exercises
exercise_counts = {
    "pushups": 0,
    "hammercurl": 0,
    "benchpress": 0,
    "lateralraises": 0,
    "bentoverrow": 0,
    "lunge": 0,
    "sideplank": 0,
    "reverse_swing": 0,
    "barbell_squat": 0,
    "situps": 0
}

# Store the current exercise
current_exercise = "pushups"

@app.route('/set_exercise', methods=['POST'])
def set_exercise():
    global current_exercise
    data = request.get_json()
    selected_exercise = data.get('exercise')

    if selected_exercise in exercise_counts:
        current_exercise = selected_exercise
        return jsonify({"status": "success", "current_exercise": current_exercise}), 200
    else:
        return jsonify({"status": "error", "message": "Invalid exercise type"}), 400

@app.route('/get_exercise_count', methods=['GET'])
def get_exercise_count():
    return jsonify({current_exercise: exercise_counts[current_exercise]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
