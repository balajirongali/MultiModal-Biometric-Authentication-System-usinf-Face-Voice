# ui_fusion_integration.py

import requests
import math
import base64
from pathlib import Path
import random


class DSFusionAuthenticator:
    """
    Implements Dempster–Shafer based fusion for face + voice evidence.
    """

    def __init__(self, face_service_url, voice_service_url):
        self.face_service_url = face_service_url
        self.voice_service_url = voice_service_url

    # ---------- Reliability from image frames ----------

    def compute_face_reliability(self, frames, num_frames):
        """
        Compute face reliability r_f ∈ [0,1] from:
        - number of frames
        - simple brightness & contrast statistics

        More frames + reasonable brightness/contrast = higher reliability.
        """
        if not frames or num_frames <= 0:
            return 0.0

        # Normalize frame count (e.g., saturate at 30 frames)
        count_factor = min(num_frames / 30.0, 1.0)

        # Simple image quality proxy (just use first frame)
        import cv2
        import numpy as np

        first = frames[0]
        gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())
        std_brightness = float(gray.std())

        # Good brightness around [60, 200]
        bright_factor = 1.0 - min(abs(mean_brightness - 128.0) / 128.0, 1.0)

        # Some contrast (std) is good, but extremely low or extremely high is bad
        contrast_norm = min(std_brightness / 64.0, 1.0)
        contrast_factor = contrast_norm

        # Combine factors
        r = 0.2 + 0.8 * (
            0.5 * count_factor + 0.25 * bright_factor + 0.25 * contrast_factor
        )
        return max(0.0, min(1.0, r))

    # ---------- Reliability from audio frames ----------

    def compute_voice_reliability(
        self, audio_frames, audio_duration, sample_rate=16000
    ):
        """
        Compute voice reliability r_v ∈ [0,1] from:
        - audio duration
        - fraction of non-silent frames
        - simple SNR-like measure
        - zero-crossing-based noise penalty

        This uses the *recorded* audio frames, so it won't be always 1.
        """
        if not audio_frames or audio_duration <= 0.0:
            return 0.0

        import numpy as np

        # Duration: saturate at 5 seconds
        duration_factor = min(audio_duration / 5.0, 1.0)

        # Analyze up to 100 chunks to keep it cheap
        num_chunks = min(len(audio_frames), 100)
        energies = []
        zcrs = []

        for chunk in audio_frames[:num_chunks]:
            if not chunk:
                continue

            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            if samples.size == 0:
                continue

            # Short-frame energy
            e = float((samples**2).mean())
            energies.append(e)

            # Zero-crossing rate (roughly: sign changes)
            signs = np.sign(samples)
            zc = float(np.mean(np.abs(np.diff(signs > 0))))
            zcrs.append(zc)

        if not energies:
            return 0.0

        energies = np.array(energies)
        zcrs = np.array(zcrs) if zcrs else np.zeros_like(energies)

        mean_energy = energies.mean()

        # Threshold between speech / non-speech
        speech_threshold = max(mean_energy * 0.3, 100.0)
        speech_frames = (energies > speech_threshold).sum()
        speech_ratio = speech_frames / float(len(energies))

        # Simple SNR-like factor: loud vs quiet parts
        quiet_energy = energies[energies <= speech_threshold]
        if quiet_energy.size > 0:
            snr_like = np.log10(
                (mean_energy + 1e-6) / (quiet_energy.mean() + 1e-6)
            )
            snr_factor = max(0.0, min(snr_like / 1.5, 1.0))
        else:
            snr_factor = 1.0

        # High ZCR everywhere suggests white noise → penalise
        avg_zcr = zcrs.mean() if zcrs.size else 0.0
        zcr_penalty = min(avg_zcr / 0.3, 1.0)  # 0 ok, 1 heavy penalty
        zcr_factor = 1.0 - 0.5 * zcr_penalty

        # Combine components (kept well away from "always 1")
        raw = (
            0.45 * duration_factor + 0.35 * speech_ratio + 0.20 * snr_factor
        ) * zcr_factor

        reliability = 0.1 + 0.9 * max(0.0, min(1.0, raw))
        return float(max(0.0, min(1.0, reliability)))

    # ---------- Mass function utilities ----------

    def score_to_mass(self, score, reliability):
        """
        Convert a model score s ∈ [0,1] and reliability r ∈ [0,1]
        into a Dempster–Shafer mass function on {G, I, Θ}:
            m(G)   = r * s
            m(I)   = r * (1 - s)
            m(Θ)   = 1 - r
        """
        s = max(0.0, min(1.0, float(score)))
        r = max(0.0, min(1.0, float(reliability)))

        mG = r * s
        mI = r * (1.0 - s)
        mU = 1.0 - r  # ignorance

        # normalize just in case of tiny numeric issues
        total = mG + mI + mU
        if total > 0:
            mG /= total
            mI /= total
            mU /= total

        return {"G": mG, "I": mI, "U": mU}

    def combine_two_sources(self, m1, m2):
        """
        Combine two mass functions using Dempster's rule.
        m1, m2: dicts with keys 'G', 'I', 'U' (for Θ).
        """
        m1G, m1I, m1U = m1["G"], m1["I"], m1["U"]
        m2G, m2I, m2U = m2["G"], m2["I"], m2["U"]

        # Conflict
        K = m1G * m2I + m1I * m2G

        if K >= 1.0:
            # Extreme conflict: return full ignorance
            return {"G": 0.0, "I": 0.0, "U": 1.0}, 1.0

        norm = 1.0 / (1.0 - K)

        mG = (m1G * m2G + m1G * m2U + m1U * m2G) * norm
        mI = (m1I * m2I + m1I * m2U + m1U * m2I) * norm
        mU = (m1U * m2U) * norm

        return {"G": mG, "I": mI, "U": mU}, K

    # ---------- Backend calling + full fusion ----------

    def call_face_service(self, frames_b64, face_threshold):
        payload = {
            "frames": frames_b64,
            "threshold": face_threshold,
        }
        resp = requests.post(
            f"{self.face_service_url}/authenticate", json=payload, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        # Expect at least: {"score": float, "identified": bool, "username": "..."}
        return data

    def call_voice_service(self, audio_b64, voice_threshold):
        payload = {
            "audio": audio_b64,
            "threshold": voice_threshold,
        }
        resp = requests.post(
            f"{self.voice_service_url}/authenticate", json=payload, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return data

    def authenticate_with_fusion(
        self,
        frames_b64,
        audio_b64,
        face_threshold,
        voice_threshold,
        face_reliability,
        voice_reliability,
    ):
        """
        Main entry point used by the UI.

        Returns dict:
            {
              "decision": bool,
              "face_evidence": {...},
              "voice_evidence": {...},
              "ds": {
                  "belief_genuine": float,
                  "plausibility_genuine": float,
                  "conflict": float,
              },
              "errors": [ ... ],
            }
        """
        result = {
            "decision": False,
            "face_evidence": {},
            "voice_evidence": {},
            "ds": {},
            "errors": [],
        }

        # ----- 1. Call both services -----
        face_data = {}
        voice_data = {}

        try:
            face_data = self.call_face_service(frames_b64, face_threshold)
        except Exception as e:
            result["errors"].append(f"Face service error: {e}")

        try:
            voice_data = self.call_voice_service(audio_b64, voice_threshold)
        except Exception as e:
            result["errors"].append(f"Voice service error: {e}")

        # Provide defaults to avoid crashes
        face_score = float(face_data.get("score", 0.0))
        voice_score = float(voice_data.get("score", 0.0))

        face_user = face_data.get("username")
        voice_user = voice_data.get("username")

        same_user = (
            face_data.get("identified")
            and voice_data.get("identified")
            and face_user
            and voice_user
            and face_user == voice_user
        )

        identity_conflict = (
            face_data.get("identified")
            and voice_data.get("identified")
            and face_user
            and voice_user
            and face_user != voice_user
        )

        if identity_conflict:
            result["errors"].append(
                f"Identity mismatch between modalities: "
                f"face={face_user}, voice={voice_user}"
            )
            # Heavy discount if the two subsystems disagree on identity.
            face_reliability *= 0.3
            voice_reliability *= 0.3

        if same_user:
            result["final_username"] = face_user

        # Save evidence details
        face_data["reliability"] = face_reliability
        voice_data["reliability"] = voice_reliability

        result["face_evidence"] = face_data
        result["voice_evidence"] = voice_data

        # ----- 2. Build mass functions -----
        m_face = self.score_to_mass(face_score, face_reliability)
        m_voice = self.score_to_mass(voice_score, voice_reliability)

        # ----- 3. DS combination -----
        m_combined, conflict = self.combine_two_sources(m_face, m_voice)

        belief_genuine = m_combined["G"]
        belief_impostor = m_combined["I"]
        plaus_genuine = 1.0 - belief_impostor  # Pl(G) = 1 - Bel(I)

        result["ds"] = {
            "m_face": m_face,
            "m_voice": m_voice,
            "m_combined": m_combined,
            "conflict": conflict,
            "belief_genuine": belief_genuine,
            "belief_impostor": belief_impostor,
            "plausibility_genuine": plaus_genuine,
            "same_user": same_user,
            "identity_conflict": identity_conflict,
        }

        # ----- 4. Decision rule -----
        # Example: accept if DS belief and plausibility are strong enough
        accept = (
            same_user
            and belief_genuine >= 0.60
            and plaus_genuine >= 0.80
            and conflict <= 0.50
        )

        result["decision"] = accept
        return result

    # ---------- Reporting helpers for UI ----------

    def get_decision_message(self, fusion_result):
        ds = fusion_result.get("ds", {})
        belief_g = ds.get("belief_genuine", 0.0)
        plaus_g = ds.get("plausibility_genuine", 0.0)
        conflict = ds.get("conflict", 0.0)

        face = fusion_result.get("face_evidence", {})
        voice = fusion_result.get("voice_evidence", {})
        user = (
            fusion_result.get("final_username")
            or face.get("username")
            or voice.get("username")
        )

        if fusion_result.get("decision", False):
            user_part = f" for user '{user}'" if user else ""
            return (
                f"Authentication successful{user_part} via DS fusion.\n"
                f"Belief(Genuine) = {belief_g:.3f}, "
                f"Plausibility(Genuine) = {plaus_g:.3f}, "
                f"Conflict = {conflict:.3f}"
            )
        else:
            user_part = f" for claimed user '{user}'" if user else ""
            return (
                f"Authentication rejected{user_part} by DS fusion.\n"
                f"Belief(Genuine) = {belief_g:.3f}, "
                f"Plausibility(Genuine) = {plaus_g:.3f}, "
                f"Conflict = {conflict:.3f}"
            )

    def get_detailed_report(self, fusion_result):
        face = fusion_result.get("face_evidence", {})
        voice = fusion_result.get("voice_evidence", {})
        ds = fusion_result.get("ds", {})

        m_face = ds.get("m_face", {})
        m_voice = ds.get("m_voice", {})
        m_comb = ds.get("m_combined", {})

        lines = []
        lines.append("Dempster–Shafer Fusion Report")
        lines.append("-" * 40)
        lines.append(
            f"Face: score={face.get('score', 0):.3f}, "
            f"reliability={face.get('reliability', 0):.3f}, "
            f"identified={face.get('identified')}"
        )
        lines.append(
            f"Voice: score={voice.get('score', 0):.3f}, "
            f"reliability={voice.get('reliability', 0):.3f}, "
            f"identified={voice.get('identified')}"
        )

        same_user = ds.get("same_user", False)
        identity_conflict = ds.get("identity_conflict", False)

        lines.append(
            f"Face identity: {face.get('username')}, "
            f"Voice identity: {voice.get('username')}"
        )
        if identity_conflict:
            lines.append("⚠ Identity mismatch between face and voice.")
        elif same_user:
            lines.append("✓ Face and voice agree on the same user.")
        lines.append("")

        lines.append("")
        lines.append(
            "Face mass function (m_f): "
            f"G={m_face.get('G', 0):.3f}, "
            f"I={m_face.get('I', 0):.3f}, "
            f"Θ={m_face.get('U', 0):.3f}"
        )
        lines.append(
            "Voice mass function (m_v): "
            f"G={m_voice.get('G', 0):.3f}, "
            f"I={m_voice.get('I', 0):.3f}, "
            f"Θ={m_voice.get('U', 0):.3f}"
        )
        lines.append("")
        lines.append(
            "Combined mass (m_fv): "
            f"G={m_comb.get('G', 0):.3f}, "
            f"I={m_comb.get('I', 0):.3f}, "
            f"Θ={m_comb.get('U', 0):.3f}"
        )
        lines.append(
            f"Bel(G) = {ds.get('belief_genuine', 0):.3f}, "
            f"Pl(G) = {ds.get('plausibility_genuine', 0):.3f}, "
            f"Conflict K = {ds.get('conflict', 0):.3f}"
        )

        if fusion_result.get("decision", False):
            lines.append("")
            lines.append("Decision: ACCEPT (Genuine user).")
        else:
            lines.append("")
            lines.append("Decision: REJECT (Not enough support for genuine).")

        return "\n".join(lines)


class UIAuthenticationHandler:
    """
    Thin wrapper used by the UI, keeps the same interface the UI expects.
    """

    def __init__(self, face_service_url, voice_service_url):
        self.authenticator = DSFusionAuthenticator(
            face_service_url, voice_service_url
        )

    def authenticate_with_fusion(
        self,
        frames_b64,
        audio_b64,
        face_threshold,
        voice_threshold,
        face_reliability,
        voice_reliability,
    ):
        return self.authenticator.authenticate_with_fusion(
            frames_b64=frames_b64,
            audio_b64=audio_b64,
            face_threshold=face_threshold,
            voice_threshold=voice_threshold,
            face_reliability=face_reliability,
            voice_reliability=voice_reliability,
        )


class BulkTester:
    """
    Command-line bulk testing using the same backend services as the UI.

    Assumes directory layout:
        data_root/faces/<user_id>/*.jpg|*.png|...
        data_root/voices/<user_id>/*.wav|*.flac|...

    Reuses UIAuthenticationHandler / DSFusionAuthenticator to call services.
    """

    def __init__(self, ui_handler: UIAuthenticationHandler):
        self.ui_handler = ui_handler
        self.auth = ui_handler.authenticator  # DSFusionAuthenticator

    # ---------- Helpers ----------

    @staticmethod
    def _cli_log(msg: str):
        from datetime import datetime

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}")

    @staticmethod
    def _file_to_b64(path: Path) -> str:
        with path.open("rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def _register_user_simple(
        self, user_id: str, face_file: Path, voice_file: Path
    ):
        """
        Very simple registration using one face image + one voice file.
        Adjust payloads if your backend expects something different.
        """
        import requests

        face_b64 = self._file_to_b64(face_file)
        audio_b64 = self._file_to_b64(voice_file)

        # ---- Face registration ----
        face_payload = {
            "username": user_id,
            "frames": [face_b64],
        }
        voice_payload = {
            "username": user_id,
            "audio": audio_b64,
        }

        # Using same base URLs as DSFusionAuthenticator
        face_url = f"{self.auth.face_service_url}/register"
        voice_url = f"{self.auth.voice_service_url}/register"

        self._cli_log(f"Registering face for {user_id} using {face_file.name}")
        r_face = requests.post(face_url, json=face_payload, timeout=60)
        r_face.raise_for_status()

        self._cli_log(
            f"Registering voice for {user_id} using {voice_file.name}"
        )
        r_voice = requests.post(voice_url, json=voice_payload, timeout=60)
        r_voice.raise_for_status()

    def _auth_pair(
        self,
        user_id: str,
        face_file: Path,
        voice_file: Path,
        face_threshold: float,
        voice_threshold: float,
    ):
        """
        Authenticate a single (face, voice) pair using the same DS fusion API
        as the UI. Returns (is_correct, predicted_user, fusion_result).
        """
        # For offline files, we don't have per-chunk reliability; use fixed values.
        face_b64 = self._file_to_b64(face_file)
        audio_b64 = self._file_to_b64(voice_file)

        fusion = self.ui_handler.authenticate_with_fusion(
            frames_b64=[face_b64],
            audio_b64=audio_b64,
            face_threshold=face_threshold,
            voice_threshold=voice_threshold,
            face_reliability=0.9,
            voice_reliability=0.9,
        )

        face_ev = fusion.get("face_evidence", {})
        voice_ev = fusion.get("voice_evidence", {})
        ds = fusion.get("ds", {})

        predicted = (
            fusion.get("final_username")
            or face_ev.get("username")
            or voice_ev.get("username")
        )

        is_correct = bool(
            predicted == user_id and fusion.get("decision", False)
        )

        self._cli_log(
            f"Test sample for {user_id}: "
            f"pred={predicted}, "
            f"decision={fusion.get('decision')}, "
            f"Bel(G)={ds.get('belief_genuine', 0):.3f}, "
            f"Pl(G)={ds.get('plausibility_genuine', 0):.3f}"
        )

        return is_correct, predicted, fusion

    def _prepare_user_splits(
        self,
        faces_root: Path,
        voices_root: Path,
        min_train_samples: int,
        max_test_samples: int,
    ):
        """
        Precompute train/test splits for all users before any registration happens.

        Returns:
          user_splits: {
             user_id: {
                "train_faces": [Path, ...],
                "train_voices": [Path, ...],
                "test_faces": [Path, ...],
                "test_voices": [Path, ...],
             },
             ...
          }
        """
        user_splits = {}

        face_users = {p.name for p in faces_root.iterdir() if p.is_dir()}
        voice_users = {p.name for p in voices_root.iterdir() if p.is_dir()}
        user_ids = sorted(face_users & voice_users)

        if not user_ids:
            self._cli_log("No common users found in faces/ and voices/.")
            return {}

        self._cli_log(
            f"Found {len(user_ids)} users with both face and voice data."
        )

        for user_id in user_ids:
            user_face_dir = faces_root / user_id
            user_voice_dir = voices_root / user_id

            face_files = sorted(
                f for f in user_face_dir.iterdir() if f.is_file()
            )
            voice_files = sorted(
                f for f in user_voice_dir.iterdir() if f.is_file()
            )

            if (
                len(face_files) < min_train_samples
                or len(voice_files) < min_train_samples
            ):
                self._cli_log(
                    f"Skipping {user_id}: not enough samples "
                    f"(faces={len(face_files)}, voices={len(voice_files)})."
                )
                continue

            random.shuffle(face_files)
            random.shuffle(voice_files)

            train_faces = face_files[:min_train_samples]
            train_voices = voice_files[:min_train_samples]

            test_faces = face_files[
                min_train_samples : min_train_samples + max_test_samples
            ]
            test_voices = voice_files[
                min_train_samples : min_train_samples + max_test_samples
            ]

            if not test_faces or not test_voices:
                self._cli_log(
                    f"Skipping tests for {user_id}: not enough leftover "
                    f"(faces={len(face_files)}, voices={len(voice_files)})."
                )
                continue

            user_splits[user_id] = {
                "train_faces": train_faces,
                "train_voices": train_voices,
                "test_faces": test_faces,
                "test_voices": test_voices,
            }

        return user_splits

    # ---------- Public entry point ----------

    def run_bulk_testing(
        self,
        data_root: str = "biometric_data",
        max_test_samples: int = 5,
        min_train_samples: int = 3,
        face_threshold: float = 0.3,
        voice_threshold: float = 0.3,
    ):
        """
        Main bulk testing loop.

        PHASE 1: prepare splits for all users.
        PHASE 2: register all users that have enough data.
        PHASE 3: test all registered users and compute accuracy.
        """
        root = Path(data_root)
        faces_root = root / "faces"
        voices_root = root / "voices"

        if not faces_root.exists() or not voices_root.exists():
            self._cli_log(
                f"Missing faces/ or voices/ in data_root={data_root}. Nothing to test."
            )
            return

        self._cli_log(
            f"min_train_samples={min_train_samples}, max_test_samples={max_test_samples}"
        )

        # --- RESET DATABASE (WIPES PREVIOUS REGISTRATIONS) ---

        for db_file in [
            "/home/yagna/Code/FYP/Face/embeddings_db.pkl",
            "/home/yagna/Code/FYP/SpeechBrain/users.json",
        ]:
            db_path = Path(db_file)
            if db_path.exists():
                db_path.unlink()
                self._cli_log(f"Deleted existing DB file → {db_file}")
            else:
                self._cli_log(f"No existing DB found → {db_file}")

        # -------- PHASE 1: prepare user splits --------
        user_splits = self._prepare_user_splits(
            faces_root=faces_root,
            voices_root=voices_root,
            min_train_samples=min_train_samples,
            max_test_samples=max_test_samples,
        )

        if not user_splits:
            self._cli_log("No valid users with enough train/test samples.")
            return

        user_ids = sorted(user_splits.keys())
        self._cli_log(f"Will process {len(user_ids)} users: {user_ids}")

        # -------- PHASE 2: registration for all users --------
        registered_users = set()
        self._cli_log("--- Starting registration for all users ---")
        for user_id in user_ids:
            splits = user_splits[user_id]
            train_faces = splits["train_faces"]
            train_voices = splits["train_voices"]

            try:
                self._register_user_simple(
                    user_id=user_id,
                    face_file=train_faces[0],
                    voice_file=train_voices[0],
                )
                registered_users.add(user_id)
            except Exception as e:
                self._cli_log(
                    f"Registration failed for {user_id}: {e}. Skipping user."
                )
                continue
        self._cli_log(
            f"Registration phase complete. Successfully registered {len(registered_users)} users."
        )

        if not registered_users:
            self._cli_log("No users registered successfully; aborting tests.")
            return

        # -------- PHASE 3: testing for all registered users --------
        total_tests = 0
        total_correct = 0

        self._cli_log("--- Starting testing phase ---")
        for user_id in sorted(registered_users):
            splits = user_splits[user_id]
            test_faces = splits["test_faces"]
            test_voices = splits["test_voices"]

            user_tests = 0
            user_correct = 0

            num_pairs = min(len(test_faces), len(test_voices))
            for i in range(num_pairs):
                face_file = test_faces[i]
                voice_file = test_voices[i]

                try:
                    ok, pred, fusion = self._auth_pair(
                        user_id=user_id,
                        face_file=face_file,
                        voice_file=voice_file,
                        face_threshold=face_threshold,
                        voice_threshold=voice_threshold,
                    )
                except Exception as e:
                    self._cli_log(
                        f"Error authenticating sample {i} for {user_id}: {e}"
                    )
                    continue

                user_tests += 1
                total_tests += 1
                if ok:
                    user_correct += 1
                    total_correct += 1

            if user_tests == 0:
                self._cli_log(f"No valid tests executed for {user_id}.")
                continue

            user_acc = 100.0 * user_correct / user_tests
            self._cli_log(
                f"User {user_id}: {user_correct}/{user_tests} correct "
                f"({user_acc:.2f}% accuracy)."
            )

        if total_tests == 0:
            self._cli_log(
                "No tests executed at all. Check your dataset / settings."
            )
        else:
            overall_acc = 100.0 * total_correct / total_tests
            self._cli_log(
                f"Overall: {total_correct}/{total_tests} correct "
                f"({overall_acc:.2f}% accuracy)."
            )
