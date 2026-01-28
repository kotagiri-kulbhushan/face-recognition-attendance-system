from flask import Flask, render_template, request, Response
from face_ops import gen_frames, register_person, recognize, current_status

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/register", methods=["POST"])
def register():
    name = request.form["name"]
    register_person(name)
    return render_template("index.html")

@app.route("/checkin")
def checkin():
    recognize("in")
    return render_template("index.html")

@app.route("/checkout")
def checkout():
    recognize("out")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
