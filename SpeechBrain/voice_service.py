"""
PART 2: Voice Recognition Service (Flask API)
Save as: voice_service.py
Run in your voice recognition repo with its venv (Python 3.7)

Requirements: pip install flask flask-cors
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import tempfile
import json

# Import your voice utilities
from voice_auth import register_user, identify_user, extract_embedding, load_db

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    logging.info("Health check requested.")
    return jsonify({"status": "ok", "service": "voice_recognition"})


@app.route("/register", methods=["POST"])
def register_voice():
    """
    Expects: {
        "username": "alice",
        "audio": "base64_encoded_wav_file"
    }
    """
    try:
        data = request.json
        username = data.get("username")
        audio_b64 = data.get("audio")
        logging.info(f"Register request received for user: {username}.")

        if not username or not audio_b64:
            logging.warning(
                "Missing username or audio in voice registration request."
            )
            return jsonify({"error": "Missing username or audio"}), 400

        # Decode and save to temp file
        audio_data = base64.b64decode(audio_b64)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(audio_data)
        temp_file.close()
        logging.debug(f"Audio saved to temporary file: {temp_file.name}")

        try:
            # Register using the temp file
            result = register_user(username, [temp_file.name])
            logging.info(
                f"Voice registration successful for user {username}: {result}"
            )

            return jsonify(
                {"success": True, "username": username, "message": result}
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
                logging.debug(f"Temporary audio file removed: {temp_file.name}")

    except Exception as e:
        logging.error(
            f"Error during voice registration for user {username}: {e}",
            exc_info=True,
        )
        return jsonify({"error": str(e)}), 500


@app.route("/authenticate", methods=["POST"])
def authenticate_voice():
    """
    Expects: {
        "audio": "base64_encoded_wav_file",
        "threshold": 0.7
    }
    """
    try:
        data = request.json
        audio_b64 = data.get("audio")
        threshold = data.get("threshold", 0.7)
        logging.info(
            f"Authenticate request received with threshold: {threshold}."
        )

        if not audio_b64:
            logging.warning(
                "No audio provided in voice authentication request."
            )
            return jsonify({"error": "No audio provided"}), 400

        # Decode and save to temp file
        audio_data = base64.b64decode(audio_b64)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(audio_data)
        temp_file.close()
        logging.debug(f"Audio saved to temporary file: {temp_file.name}")

        try:
            # Identify using the temp file
            result = identify_user(temp_file.name, threshold=threshold)
            logging.info(f"Voice identification result: {result}")

            # Parse result string
            if "IDENTIFIED" in result:
                # Extract username and score
                parts = result.split(":")
                if len(parts) >= 2:
                    user_info = parts[1].strip()
                    username = user_info.split("(")[0].strip()
                    score_str = user_info.split("score=")[1].rstrip(")")
                    score = float(score_str)
                    logging.info(
                        f"Voice authentication successful for user: {username} with score: {score:.3f}"
                    )

                    return jsonify(
                        {
                            "success": True,
                            "identified": True,
                            "username": username,
                            "score": score,
                            "message": result,
                        }
                    )

            # Not identified
            logging.info("Voice authentication: No user identified.")
            return jsonify(
                {"success": True, "identified": False, "message": result}
            )

        finally:
            # Clean up temp file
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
                logging.debug(f"Temporary audio file removed: {temp_file.name}")

    except Exception as e:
        logging.error(f"Error during voice authentication: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/users", methods=["GET"])
def list_users():
    """List all registered users"""
    try:
        logging.info("Request to list registered users received.")
        db = load_db()
        users = list(db.keys())
        logging.info(f"Returning {len(users)} registered users.")
        return jsonify({"success": True, "users": users, "count": len(db)})
    except Exception as e:
        logging.error(f"Error listing users: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🎤 Starting Voice Recognition Service on port 5002")
    app.run(host="0.0.0.0", port=5002, debug=False)
