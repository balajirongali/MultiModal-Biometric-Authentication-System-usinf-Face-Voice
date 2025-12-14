# face_utils.py
"""
Helpers for face embedding, DB management, and similarity checks.
Compatible with Python 3.10 - 3.13.
"""

import os
import pickle
import numpy as np
from typing import Dict, Optional, Tuple, List
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from PIL import Image
import cv2

EMBEDDING_DB = "embeddings_db.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models once
_mtcnn = None
_resnet = None

def load_models(device: Optional[torch.device] = None):
    global _mtcnn, _resnet
    if device is None:
        device = DEVICE
    if _mtcnn is None:
        _mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device)
    if _resnet is None:
        # pretrained on VGGFace2 is good for face verification
        _resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return _mtcnn, _resnet

def image_to_pil(img_path_or_array) -> Image.Image:
    # Accept path or numpy array (BGR from cv2) or PIL
    if isinstance(img_path_or_array, Image.Image):
        return img_path_or_array
    if isinstance(img_path_or_array, str):
        return Image.open(img_path_or_array).convert('RGB')
    # numpy array: convert BGR->RGB
    if isinstance(img_path_or_array, np.ndarray):
        rgb = cv2.cvtColor(img_path_or_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    raise ValueError("Unsupported image input type")

def get_face_embedding(img_input) -> Optional[np.ndarray]:
    """
    Returns a 512-d numpy float32 embedding for the *largest* face detected,
    or None if no face is found.
    """
    mtcnn, resnet = load_models()
    pil = image_to_pil(img_input)
    # mtcnn returns cropped face Tensor (C,H,W) normalized
    face_tensor = _mtcnn(pil)
    if face_tensor is None:
        return None
    # If mtcnn returns single tensor, shape (3,160,160)
    if isinstance(face_tensor, torch.Tensor):
        face_tensor = face_tensor.unsqueeze(0)  # make batch dim
    with torch.no_grad():
        emb = _resnet(face_tensor.to(DEVICE)).cpu().numpy()
    emb = emb[0].astype(np.float32)  # 512-d
    # Normalize to unit vector for cosine-friendly comparisons
    emb = emb / np.linalg.norm(emb)
    return emb

# DB operations
def load_db(path: str = EMBEDDING_DB) -> Dict[str, np.ndarray]:
    if os.path.exists(path):
        with open(path, 'rb') as f:
            db = pickle.load(f)
        # Ensure numpy arrays
        for k, v in list(db.items()):
            db[k] = np.array(v, dtype=np.float32)
        return db
    return {}

def save_db(db: Dict[str, np.ndarray], path: str = EMBEDDING_DB):
    with open(path, 'wb') as f:
        pickle.dump({k: v.tolist() for k, v in db.items()}, f)

def add_user_embedding(name: str, embedding: np.ndarray, db_path: str = EMBEDDING_DB, avg: bool = True):
    db = load_db(db_path)
    if name in db and avg:
        # average existing embedding with new one (keeps DB compact)
        db[name] = (db[name] + embedding) / 2.0
        db[name] = db[name] / np.linalg.norm(db[name])
    else:
        db[name] = embedding / np.linalg.norm(embedding)
    save_db(db, db_path)

# Identification
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # inputs expected unit-normalized, but safe-guard
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def identify(embedding: np.ndarray, db: Dict[str, np.ndarray], threshold: float = 0.6) -> Tuple[Optional[str], float]:
    """
    Identify by cosine similarity. Returns (best_name_or_None, best_score).
    Score is cosine similarity in [-1,1]; typical FaceNet matches > ~0.5-0.8.
    """
    if len(db) == 0:
        return None, 0.0
    best_name = None
    best_score = -1.0
    for name, db_emb in db.items():
        sim = cosine_similarity(embedding, db_emb)
        if sim > best_score:
            best_score = sim
            best_name = name
    if best_score >= threshold:
        return best_name, best_score
    return None, best_score

# Convenience: build embedding from a set of images (e.g., capture multiple shots)
def embeddings_from_images(image_list: List, skip_missing: bool = True) -> List[np.ndarray]:
    embs = []
    for img in image_list:
        e = get_face_embedding(img)
        if e is None:
            if not skip_missing:
                raise RuntimeError("No face detected in one of the images")
            continue
        embs.append(e)
    return embs

def mean_embedding(embs: List[np.ndarray]) -> Optional[np.ndarray]:
    if not embs:
        return None
    m = np.mean(np.stack(embs, axis=0), axis=0)
    return (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)
