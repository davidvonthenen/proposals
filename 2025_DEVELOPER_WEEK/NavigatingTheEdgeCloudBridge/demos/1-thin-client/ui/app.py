import multiprocessing
import warnings
from pathlib import Path


from flask import Flask, send_from_directory

# Disable future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

app = Flask(__name__, static_folder="./public", static_url_path="/public")

@app.route("/<path:filename>")
def serve_others(filename):
    return send_from_directory(app.static_folder, filename)

@app.route("/assets/<path:filename>")
def serve_image(filename):
    return send_from_directory(app.static_folder, "assets/" + filename)

@app.route("/", methods=["GET"])
def serve_index():
    return app.send_static_file("index.html")

def run_ui():
    app.run(port=5000, debug=True, use_reloader=False)

if __name__ == "__main__":
    p_flask = multiprocessing.Process(target=run_ui)
    p_flask.start()
    p_flask.join()
