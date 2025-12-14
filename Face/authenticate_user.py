# authenticate_user.py
"""
Authenticate a live user via webcam or from image.
Usage:
python authenticate_user.py --webcam
python authenticate_user.py --image try.jpg
"""

import argparse
import cv2
from face_utils import get_face_embedding, load_db, identify
import time

def run_live_auth(db_path="embeddings_db.pkl", threshold=0.6):
    db = load_db(db_path)
    if not db:
        print("Embeddings DB is empty. Register users first.")
        return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Could not open webcam")
    print("Press 'q' to quit. Press 'c' to capture/frame-check and authenticate.")
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue
            display = frame.copy()
            cv2.putText(display, "Press 'c' to authenticate", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Authenticate", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                # capture current frame and try to verify
                emb = get_face_embedding(frame)
                if emb is None:
                    print("No face detected. Try again.")
                    continue
                name, score = identify(emb, db, threshold=threshold)
                if name:
                    print(f"Authenticated as {name} (score={score:.3f})")
                else:
                    print(f"No match (best score {score:.3f}). Access denied.")
    finally:
        cam.release()
        cv2.destroyAllWindows()

def run_image_auth(image_path, db_path="embeddings_db.pkl", threshold=0.6):
    db = load_db(db_path)
    if not db:
        print("Embeddings DB empty. Register first.")
        return
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image.")
        return
    emb = get_face_embedding(img)
    if emb is None:
        print("No face found in image.")
        return
    name, score = identify(emb, db, threshold=threshold)
    if name:
        print(f"Authenticated as {name} (score={score:.3f})")
    else:
        print(f"No match (best score {score:.3f}). Access denied.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam", action="store_true", help="Use webcam to authenticate (live capture)")
    parser.add_argument("--image", help="Image file to authenticate")
    parser.add_argument("--db", default="embeddings_db.pkl")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    if args.webcam:
        run_live_auth(db_path=args.db, threshold=args.threshold)
    elif args.image:
        run_image_auth(args.image, db_path=args.db, threshold=args.threshold)
    else:
        parser.error("Specify --webcam or --image path")

if __name__ == "__main__":
    main()
