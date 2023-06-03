from flask import Flask, jsonify
app = Flask(__name__)

emotions = [
    {
        "emotion": "Angry",
        "id": 1
    },
    {
        "emotion": "Disgust",
        "id": 2
    },
    {
        "emotion": "Fear",
        "id": 3
    },
    {
        "emotion": "Happy",
        "id": 4
    },
    {
        "emotion": "Neutral",
        "id": 5
    },
    {
        "emotion": "Sad",
        "id": 6
    },
    {
        "emotion": "Surprise",
        "id": 7
    }
]

detect = [
    {
        "detect": "Stand",
        "id": 1
    },
    {
        "detect": "Fall",
        "id": 2
    }
]


@app.route('/emotions', methods=['GET'])
def displayEmotions():
    return jsonify(emotions)


@app.route('/detect', methods=['GET'])
def displayDetection():
    return (detect)


if __name__ == '__main__':
    app.run()
app.run(port=5000)
