import os
import json
import torch
from speechbrain.pretrained import SpeakerRecognition

DB_FILE = "users.json"
ECAPA_PATH = os.path.abspath("pretrained_models/ecapa")


# -----------------------------------------------------------
# Utility: Load / Save database
# -----------------------------------------------------------
def load_db():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as f:
            json.dump({}, f)
    with open(DB_FILE, "r") as f:
        return json.load(f)


def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=4)


# -----------------------------------------------------------
# Load ECAPA model
# -----------------------------------------------------------
speaker_model = SpeakerRecognition.from_hparams(
    source=ECAPA_PATH,  # absolute path
    savedir=ECAPA_PATH,  # absolute path
    run_opts={"device": "cpu"},
    overrides={},
)


# -----------------------------------------------------------
# Extract ECAPA embedding
# -----------------------------------------------------------
def extract_embedding(audio_file):
    signal = speaker_model.load_audio(audio_file)
    emb = speaker_model.encode_batch(signal)
    emb = torch.nn.functional.normalize(emb, p=2, dim=2)
    return emb.squeeze().tolist()


# -----------------------------------------------------------
# Multi-sample enrollment (average embeddings)
# -----------------------------------------------------------
def register_user(username, audio_files):
    db = load_db()
    embeddings = []

    for f in audio_files:
        emb = torch.tensor(extract_embedding(f))
        embeddings.append(emb)

    avg_emb = torch.stack(embeddings).mean(dim=0)
    db[username] = {"embedding": avg_emb.tolist()}

    save_db(db)
    return f"✔ Registered {username} with {len(audio_files)} samples."


# -----------------------------------------------------------
# Identify speaker across database
# -----------------------------------------------------------
def identify_user(audio_file, threshold=0.7):
    db = load_db()
    if not db:
        return "❌ No users registered."

    test_emb = torch.tensor(extract_embedding(audio_file)).unsqueeze(0)

    best_user = None
    best_score = -1

    for user, data in db.items():
        stored_emb = torch.tensor(data["embedding"]).unsqueeze(0)
        score = torch.nn.functional.cosine_similarity(
            stored_emb, test_emb
        ).item()

        if score > best_score:
            best_score = score
            best_user = user

    if best_score >= threshold:
        return f"✔ IDENTIFIED: {best_user} (score={best_score:.3f})"
    else:
        return f"❌ UNKNOWN speaker (best score={best_score:.3f})"


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voice Authentication System")

    parser.add_argument(
        "--register",
        nargs="+",
        help="Usage: --register username file1.wav file2.wav ...",
    )
    parser.add_argument("--identify", help="Usage: --identify file.wav")

    args = parser.parse_args()

    if args.register and len(args.register) >= 2:
        username = args.register[0]
        files = args.register[1:]
        print(register_user(username, files))

    if args.identify:
        print(identify_user(args.identify))
