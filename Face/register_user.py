# register_user.py
"""
Register a new user. Options:
 - capture N images from webcam and compute average embedding
 - or pass one or more image file paths

Usage examples:
python register_user.py --name alice --webcam --count 8
python register_user.py --name bob --images path1.jpg path2.jpg
"""

import argparse
import cv2
import time
from face_utils import get_face_embedding, embeddings_from_images, mean_embedding, add_user_embedding, load_db, save_db

def capture_images_from_webcam(count=6, delay=0.7):
    cam = cv2.VideoCapture(0)
    captured = []
    if not cam.isOpened():
        raise RuntimeError("Could not open webcam (cv2.VideoCapture(0)).")
    print("Look at the camera. Capturing images in 2 seconds...")
    time.sleep(2.0)
    taken = 0
    try:
        while taken < count:
            ret, frame = cam.read()
            if not ret:
                continue
            cv2.putText(frame, f"Capturing {taken+1}/{count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Register - Press q to cancel", frame)
            key = cv2.waitKey(1) & 0xFF
            # Automatically capture if face roughly centered (we still rely on embedding extraction)
            if key == ord('q'):
                break
            # simple timed capture
            time.sleep(delay)
            captured.append(frame.copy())
            taken += 1
        cv2.destroyAllWindows()
    finally:
        cam.release()
    return captured

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="User name to register (unique key)")
    parser.add_argument("--webcam", action="store_true", help="Capture images from webcam")
    parser.add_argument("--count", type=int, default=6, help="Number of frames to capture from webcam")
    parser.add_argument("--images", nargs='*', help="One or more image file paths to use for registration")
    parser.add_argument("--db", default="embeddings_db.pkl", help="Path to embeddings DB")
    parser.add_argument("--threshold", type=float, default=0.6, help="Optional verify threshold check against existing DB")
    args = parser.parse_args()

    imgs = []
    if args.webcam:
        imgs = capture_images_from_webcam(count=args.count)
    elif args.images:
        # load image files as numpy arrays with cv2
        for p in args.images:
            img = cv2.imread(p)
            if img is None:
                print(f"Warning: failed to read {p}; skipping")
                continue
            imgs.append(img)
    else:
        parser.error("Specify --webcam or --images ...")

    embs = embeddings_from_images(imgs)
    if not embs:
        print("No faces detected in provided images. Try again with clearer images or better lighting.")
        return

    emb = mean_embedding(embs)
    if emb is None:
        print("Failed to compute embedding.")
        return

    # Optional: check against DB to warn if this person appears similar to existing user (avoids accidental duplicate)
    from face_utils import load_db, cosine_similarity
    db = load_db(args.db)
    if db:
        best_name, best_score = None, -1
        for n, e in db.items():
            s = cosine_similarity(emb, e)
            if s > best_score:
                best_name, best_score = n, s
        print(f"Best match in DB: {best_name} (score={best_score:.3f})")
        if best_score >= args.threshold:
            print("Warning: the new embedding is very similar to an existing user. Consider choosing another name or verify identity.")

    # finally add to DB
    add_user_embedding(args.name, emb, db_path=args.db)
    print(f"Registered user '{args.name}' successfully. DB saved to {args.db}")

if __name__ == "__main__":
    main()
