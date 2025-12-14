"""
PART 1: Face Recognition Service (Flask API)
Save as: face_service.py
Run in your face recognition repo with its venv

Requirements: pip install flask flask-cors opencv-python pillow numpy
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import os

# Import your face utilities
from face_utils import (
    get_face_embedding,
    embeddings_from_images,
    mean_embedding,
    add_user_embedding,
    load_db,
    identify,
)

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    logging.info("Health check requested.")
    return jsonify({"status": "ok", "service": "face_recognition"})


@app.route("/register", methods=["POST"])
def register_face():
    """
    Expects: {
        "username": "alice",
        "frames": ["base64_image1", "base64_image2", ...]
    }
    """
    try:
        data = request.json
        username = data.get("username")
        frames_b64 = data.get("frames", [])
        logging.info(f"Register request received for user: {username} with {len(frames_b64)} frames.")

        if not username or not frames_b64:
            logging.warning("Missing username or frames in registration request.")
            return jsonify({"error": "Missing username or frames"}), 400

        # Decode frames
        images = []
        for frame_b64 in frames_b64:
            img_data = base64.b64decode(frame_b64)
            img = Image.open(BytesIO(img_data))
            img_np = np.array(img)
            # Convert RGB to BGR for OpenCV compatibility
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            images.append(img_np)

        # Extract embeddings
        face_embs = embeddings_from_images(images, skip_missing=True)

        if not face_embs:
            logging.warning(f"No faces detected in frames for user: {username}.")
            return jsonify({"error": "No faces detected in frames"}), 400

        # Average and store
        avg_emb = mean_embedding(face_embs)
        add_user_embedding(username, avg_emb)
        logging.info(f"Successfully registered user: {username} with {len(face_embs)} faces detected.")

        return jsonify(
            {
                "success": True,
                "username": username,
                "faces_detected": len(face_embs),
                "total_frames": len(images),
            }
        )

    except Exception as e:
        logging.error(f"Error during face registration for user {username}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/authenticate", methods=["POST"])
def authenticate_face():
    """
    Expects: {
        "frames": ["base64_image1", "base64_image2", ...],
        "threshold": 0.6
    }
    """
    try:
        data = request.json
        frames_b64 = data.get("frames", [])
        threshold = data.get("threshold", 0.6)
        logging.info(f"Authenticate request received with {len(frames_b64)} frames and threshold: {threshold}.")

        if not frames_b64:
            logging.warning("No frames provided in authentication request.")
            return jsonify({"error": "No frames provided"}), 400

        db = load_db()
        if not db:
            logging.warning("No users registered in face database.")
            return jsonify({"error": "No users registered"}), 400

        # Decode frames
        images = []
        for frame_b64 in frames_b64:
            img_data = base64.b64decode(frame_b64)
            img = Image.open(BytesIO(img_data))
            img_np = np.array(img)
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            images.append(img_np)

        # Identify from multiple frames
        results = []
        for img in images:
            emb = get_face_embedding(img)
            if emb is not None:
                name, score = identify(emb, db, threshold=threshold)
                if name:
                    results.append({"name": name, "score": float(score)})
                    logging.debug(f"Face identified: {name} with score {score:.3f}")
                else:
                    logging.debug("No face identified in a frame.")
            else:
                logging.debug("No face embedding extracted from a frame.")

        if not results:
            logging.info("No faces detected or matched during authentication.")
            return jsonify(
                {
                    "success": False,
                    "identified": False,
                    "message": "No faces detected or matched",
                }
            )

        # Find most common match
        from collections import Counter

        names = [r["name"] for r in results]
        most_common = Counter(names).most_common(1)[0]
        best_name = most_common[0]
        count = most_common[1]
        avg_score = (
            sum(r["score"] for r in results if r["name"] == best_name) / count
        )
        logging.info(f"Face authentication successful for user: {best_name} with average score: {avg_score:.3f} from {count} matches.")

        return jsonify(
            {
                "success": True,
                "identified": True,
                "username": best_name,
                "score": avg_score,
                "match_count": count,
                "total_frames": len(images),
            }
        )

    except Exception as e:
        logging.error(f"Error during face authentication: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🔷 Starting Face Recognition Service on port 5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
