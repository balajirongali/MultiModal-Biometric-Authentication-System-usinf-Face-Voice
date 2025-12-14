"""
PART 3: UI Client (Works with any Python 3.x)
Save as: biometric_ui.py
Run in a separate directory with its own venv

Requirements: pip install opencv-python pillow pyaudio requests
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
import pyaudio
import wave
import os
import tempfile
import time
import base64
import requests
from datetime import datetime
from io import BytesIO
from ui_fusion_integration import UIAuthenticationHandler

# Service URLs
FACE_SERVICE_URL = "http://localhost:5001"
VOICE_SERVICE_URL = "http://localhost:5002"


class BiometricAuthUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Biometric Authentication System (Microservices)")
        self.root.geometry("1000x700")

        # Video capture variables
        self.cap = None
        self.current_camera = 0
        self.is_recording = False
        self.recorded_frames = []
        self.video_thread = None
        self.stop_video = False

        # Audio recording variables
        self.audio = pyaudio.PyAudio()
        self.audio_stream = None
        self.audio_frames = []
        self.is_recording_audio = False
        self.audio_thread = None
        self.current_mic = 0

        # Temp files
        self.temp_audio_files = []

        # UI State
        self.camera_devices = self.enumerate_cameras()
        self.mic_devices = self.enumerate_microphones()

        # Create UI
        self.create_widgets()

        # Check services on startup
        self.root.after(500, self.check_services)

        # Start video after UI is ready
        self.root.after(1000, self.start_video_preview)

        # Bind close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.auth_handler = UIAuthenticationHandler(
            FACE_SERVICE_URL, VOICE_SERVICE_URL
        )

    def check_services(self):
        """Check if backend services are running"""
        services = {
            "Face Recognition": FACE_SERVICE_URL,
            "Voice Recognition": VOICE_SERVICE_URL,
        }

        status_msg = "Service Status:\n"
        all_ok = True

        for name, url in services.items():
            try:
                response = requests.get(f"{url}/health", timeout=2)
                if response.status_code == 200:
                    status_msg += f"✓ {name}: OK\n"
                else:
                    status_msg += f"✗ {name}: Error\n"
                    all_ok = False
            except:
                status_msg += f"✗ {name}: Not running\n"
                all_ok = False

        if all_ok:
            self.service_status_label.config(
                text="All services OK", foreground="green"
            )
        else:
            self.service_status_label.config(
                text="Some services offline", foreground="red"
            )
            messagebox.showwarning(
                "Service Warning",
                status_msg + "\nSome services are not available. "
                "Please start them before using the application.",
            )

    def enumerate_cameras(self):
        """Find available cameras"""
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == "nt" else 0)
            if cap is not None and cap.isOpened():
                cameras.append(i)
                cap.release()
        return cameras if cameras else [0]

    def enumerate_microphones(self):
        """Find available microphones"""
        mics = []
        try:
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    mics.append((i, info.get("name", f"mic-{i}")))
        except Exception:
            # fallback to default mic
            mics = [(0, "Default")]
        return mics if mics else [(0, "Default")]

    def create_widgets(self):
        """Create the main UI layout"""
        # Service status bar
        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill="x", padx=5, pady=2)

        ttk.Button(
            status_bar, text="🔄 Check Services", command=self.check_services
        ).pack(side="left", padx=5)
        self.service_status_label = ttk.Label(
            status_bar, text="Ready", foreground="green"
        )
        self.service_status_label.pack(side="left", padx=10)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Registration Tab
        self.register_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.register_frame, text="Registration")
        self.create_registration_tab()

        # Authentication Tab
        self.auth_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.auth_frame, text="Authentication")
        self.create_authentication_tab()

    def create_registration_tab(self):
        """Create registration interface"""
        # Top controls
        control_frame = ttk.Frame(self.register_frame)
        control_frame.pack(fill="x", padx=10, pady=5)

        # Username
        ttk.Label(control_frame, text="Username:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.reg_username = ttk.Entry(control_frame, width=20)
        self.reg_username.grid(row=0, column=1, padx=5, pady=5)

        # Camera selection
        ttk.Label(control_frame, text="Camera:").grid(
            row=0, column=2, padx=5, pady=5
        )
        self.reg_camera_var = tk.IntVar(value=self.current_camera)
        self.reg_camera_combo = ttk.Combobox(
            control_frame,
            textvariable=self.reg_camera_var,
            values=self.camera_devices,
            width=10,
            state="readonly",
        )
        self.reg_camera_combo.grid(row=0, column=3, padx=5, pady=5)
        self.reg_camera_combo.bind(
            "<<ComboboxSelected>>", self.on_camera_change
        )

        # Microphone selection
        ttk.Label(control_frame, text="Microphone:").grid(
            row=0, column=4, padx=5, pady=5
        )
        mic_names = [f"{i}: {name[:40]}" for i, name in self.mic_devices]
        self.reg_mic_var = tk.StringVar(value=mic_names[0] if mic_names else "")
        self.reg_mic_combo = ttk.Combobox(
            control_frame,
            textvariable=self.reg_mic_var,
            values=mic_names,
            width=35,
            state="readonly",
        )
        self.reg_mic_combo.grid(row=0, column=5, padx=5, pady=5)

        # Video preview
        video_frame = ttk.LabelFrame(self.register_frame, text="Video Preview")
        video_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.reg_video_label = tk.Label(video_frame, bg="black")
        self.reg_video_label.pack(fill="both", expand=True)

        # Recording controls
        rec_control_frame = ttk.Frame(self.register_frame)
        rec_control_frame.pack(fill="x", padx=10, pady=5)

        self.reg_record_btn = ttk.Button(
            rec_control_frame,
            text="▶ Start Recording",
            command=self.start_recording,
        )
        self.reg_record_btn.pack(side="left", padx=5)

        self.reg_stop_btn = ttk.Button(
            rec_control_frame,
            text="⏹ Stop Recording",
            command=self.stop_recording,
            state="disabled",
        )
        self.reg_stop_btn.pack(side="left", padx=5)

        self.reg_clear_btn = ttk.Button(
            rec_control_frame, text="🗑 Clear", command=self.clear_recording
        )
        self.reg_clear_btn.pack(side="left", padx=5)

        # Status
        self.reg_status_label = ttk.Label(
            rec_control_frame, text="Ready", foreground="green"
        )
        self.reg_status_label.pack(side="left", padx=20)

        # Frame count
        self.reg_frame_count_label = ttk.Label(
            rec_control_frame, text="Frames: 0 | Audio: 0s"
        )
        self.reg_frame_count_label.pack(side="left", padx=5)

        # Register button
        register_btn_frame = ttk.Frame(self.register_frame)
        register_btn_frame.pack(fill="x", padx=10, pady=5)

        self.reg_register_btn = ttk.Button(
            register_btn_frame,
            text="✓ Register User",
            command=self.register_user,
            state="disabled",
        )
        self.reg_register_btn.pack(side="left", padx=5)

        # Log output
        log_frame = ttk.LabelFrame(self.register_frame, text="Log")
        log_frame.pack(fill="both", expand=False, padx=10, pady=5)

        self.reg_log = scrolledtext.ScrolledText(
            log_frame, height=6, wrap=tk.WORD
        )
        self.reg_log.pack(fill="both", expand=True, padx=5, pady=5)

    def create_authentication_tab(self):
        """Create authentication interface"""
        # Top controls
        control_frame = ttk.Frame(self.auth_frame)
        control_frame.pack(fill="x", padx=10, pady=5)

        # Camera selection
        ttk.Label(control_frame, text="Camera:").grid(
            row=0, column=0, padx=5, pady=5
        )
        self.auth_camera_var = tk.IntVar(value=self.current_camera)
        self.auth_camera_combo = ttk.Combobox(
            control_frame,
            textvariable=self.auth_camera_var,
            values=self.camera_devices,
            width=10,
            state="readonly",
        )
        self.auth_camera_combo.grid(row=0, column=1, padx=5, pady=5)
        self.auth_camera_combo.bind(
            "<<ComboboxSelected>>", self.on_camera_change
        )

        # Microphone selection
        ttk.Label(control_frame, text="Microphone:").grid(
            row=0, column=2, padx=5, pady=5
        )
        mic_names = [f"{i}: {name[:40]}" for i, name in self.mic_devices]
        self.auth_mic_var = tk.StringVar(
            value=mic_names[0] if mic_names else ""
        )
        self.auth_mic_combo = ttk.Combobox(
            control_frame,
            textvariable=self.auth_mic_var,
            values=mic_names,
            width=35,
            state="readonly",
        )
        self.auth_mic_combo.grid(row=0, column=3, padx=5, pady=5)

        # Threshold
        ttk.Label(control_frame, text="Face Threshold:").grid(
            row=0, column=4, padx=5, pady=5
        )
        self.auth_face_threshold = ttk.Scale(
            control_frame, from_=0.3, to=0.9, orient="horizontal", length=120
        )
        self.auth_face_threshold.set(0.6)
        self.auth_face_threshold.grid(row=0, column=5, padx=5, pady=5)

        self.auth_face_threshold_label = ttk.Label(control_frame, text="0.60")
        self.auth_face_threshold_label.grid(row=0, column=6, padx=2, pady=5)
        self.auth_face_threshold.config(
            command=lambda v: self.auth_face_threshold_label.config(
                text=f"{float(v):.2f}"
            )
        )

        # Voice threshold
        ttk.Label(control_frame, text="Voice Threshold:").grid(
            row=1, column=4, padx=5, pady=5
        )
        self.auth_voice_threshold = ttk.Scale(
            control_frame, from_=0.3, to=0.9, orient="horizontal", length=120
        )
        self.auth_voice_threshold.set(0.7)
        self.auth_voice_threshold.grid(row=1, column=5, padx=5, pady=5)

        self.auth_voice_threshold_label = ttk.Label(control_frame, text="0.70")
        self.auth_voice_threshold_label.grid(row=1, column=6, padx=2, pady=5)
        self.auth_voice_threshold.config(
            command=lambda v: self.auth_voice_threshold_label.config(
                text=f"{float(v):.2f}"
            )
        )

        # Video preview
        video_frame = ttk.LabelFrame(self.auth_frame, text="Video Preview")
        video_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.auth_video_label = tk.Label(video_frame, bg="black")
        self.auth_video_label.pack(fill="both", expand=True)

        # Auth controls
        auth_control_frame = ttk.Frame(self.auth_frame)
        auth_control_frame.pack(fill="x", padx=10, pady=5)

        self.auth_record_btn = ttk.Button(
            auth_control_frame,
            text="▶ Start Recording",
            command=self.start_recording,
        )
        self.auth_record_btn.pack(side="left", padx=5)

        self.auth_stop_btn = ttk.Button(
            auth_control_frame,
            text="⏹ Stop Recording",
            command=self.stop_recording,
            state="disabled",
        )
        self.auth_stop_btn.pack(side="left", padx=5)

        self.auth_clear_btn = ttk.Button(
            auth_control_frame, text="🗑 Clear", command=self.clear_recording
        )
        self.auth_clear_btn.pack(side="left", padx=5)

        # Status
        self.auth_status_label = ttk.Label(
            auth_control_frame, text="Ready", foreground="green"
        )
        self.auth_status_label.pack(side="left", padx=20)

        # Frame count
        self.auth_frame_count_label = ttk.Label(
            auth_control_frame, text="Frames: 0 | Audio: 0s"
        )
        self.auth_frame_count_label.pack(side="left", padx=5)

        # Authenticate button
        auth_btn_frame = ttk.Frame(self.auth_frame)
        auth_btn_frame.pack(fill="x", padx=10, pady=5)

        self.auth_authenticate_btn = ttk.Button(
            auth_btn_frame,
            text="🔓 Authenticate",
            command=self.authenticate_user,
            state="disabled",
        )
        self.auth_authenticate_btn.pack(side="left", padx=5)

        # Log output
        log_frame = ttk.LabelFrame(self.auth_frame, text="Authentication Log")
        log_frame.pack(fill="both", expand=False, padx=10, pady=5)

        self.auth_log = scrolledtext.ScrolledText(
            log_frame, height=6, wrap=tk.WORD
        )
        self.auth_log.pack(fill="both", expand=True, padx=5, pady=5)

    def on_camera_change(self, event=None):
        """Handle camera selection change"""
        widget = event.widget
        try:
            new_camera = int(widget.get())
        except Exception:
            return
        if new_camera != self.current_camera:
            self.current_camera = new_camera
            self.restart_video_preview()

    def start_video_preview(self):
        """Start video preview thread"""
        self.stop_video = False
        self.video_thread = threading.Thread(
            target=self.video_loop, daemon=True
        )
        self.video_thread.start()

    def restart_video_preview(self):
        """Restart video with new camera"""
        self.stop_video = True
        if self.video_thread:
            self.video_thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        time.sleep(0.5)
        self.start_video_preview()

    def video_loop(self):
        """Main video capture loop"""
        # Try to open camera; on some systems specifying API preference helps, but keep generic
        self.cap = cv2.VideoCapture(self.current_camera)
        # prefer a modest resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while not self.stop_video:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Store frame if recording
            if self.is_recording:
                self.recorded_frames.append(frame.copy())

            # Convert to PhotoImage for display
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image=img)
            except Exception:
                time.sleep(0.03)
                continue

            # Update appropriate label based on active tab
            try:
                current_tab = self.notebook.index(self.notebook.select())
                if current_tab == 0:  # Registration
                    self.reg_video_label.config(image=photo)
                    self.reg_video_label.image = photo
                else:  # Authentication
                    self.auth_video_label.config(image=photo)
                    self.auth_video_label.image = photo
            except Exception:
                pass

            time.sleep(0.03)  # ~30 FPS

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

    def start_recording(self):
        """Start recording video and audio"""
        # ensure video thread running
        if self.cap is None:
            messagebox.showwarning("Warning", "Camera not ready yet.")
            return

        self.is_recording = True
        self.recorded_frames = []
        self.audio_frames = []

        # Update UI
        current_tab = self.notebook.index(self.notebook.select())
        if current_tab == 0:
            self.reg_record_btn.config(state="disabled")
            self.reg_stop_btn.config(state="normal")
            self.reg_status_label.config(text="Recording...", foreground="red")
        else:
            self.auth_record_btn.config(state="disabled")
            self.auth_stop_btn.config(state="normal")
            self.auth_status_label.config(text="Recording...", foreground="red")

        # Start audio recording thread
        self.is_recording_audio = True
        self.audio_thread = threading.Thread(
            target=self.record_audio, daemon=True
        )
        self.audio_thread.start()

        # Start frame counter update
        self.update_frame_count()

    def record_audio(self):
        """Record audio in background thread"""
        # Get selected mic index
        current_tab = self.notebook.index(self.notebook.select())
        if current_tab == 0:
            mic_str = self.reg_mic_var.get()
        else:
            mic_str = self.auth_mic_var.get()

        mic_idx = 0
        try:
            mic_idx = int(mic_str.split(":")[0]) if ":" in mic_str else 0
        except Exception:
            mic_idx = 0

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        try:
            self.audio_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=mic_idx,
                frames_per_buffer=CHUNK,
            )

            while self.is_recording_audio:
                try:
                    data = self.audio_stream.read(
                        CHUNK, exception_on_overflow=False
                    )
                    self.audio_frames.append(data)
                except Exception as e:
                    print(f"Audio read error: {e}")
                    break

        except Exception as e:
            self.log_message(
                f"Audio recording error: {e}",
                is_registration=(current_tab == 0),
            )
        finally:
            if self.audio_stream:
                try:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                except Exception:
                    pass

    def update_frame_count(self):
        """Update frame and audio duration display"""
        if self.is_recording:
            frame_count = len(self.recorded_frames)
            audio_duration = len(self.audio_frames) * 1024 / 16000

            current_tab = self.notebook.index(self.notebook.select())
            if current_tab == 0:
                self.reg_frame_count_label.config(
                    text=f"Frames: {frame_count} | Audio: {audio_duration:.1f}s"
                )
            else:
                self.auth_frame_count_label.config(
                    text=f"Frames: {frame_count} | Audio: {audio_duration:.1f}s"
                )

            self.root.after(100, self.update_frame_count)

    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        self.is_recording_audio = False

        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)

        # Update UI
        current_tab = self.notebook.index(self.notebook.select())
        if current_tab == 0:
            self.reg_record_btn.config(state="normal")
            self.reg_stop_btn.config(state="disabled")
            self.reg_status_label.config(
                text="Recording stopped", foreground="orange"
            )
            # only enable register when both video+audio exist
            if self.recorded_frames and self.audio_frames:
                self.reg_register_btn.config(state="normal")
        else:
            self.auth_record_btn.config(state="normal")
            self.auth_stop_btn.config(state="disabled")
            self.auth_status_label.config(
                text="Recording stopped", foreground="orange"
            )
            if self.recorded_frames and self.audio_frames:
                self.auth_authenticate_btn.config(state="normal")

        self.log_message(
            f"Captured {len(self.recorded_frames)} frames and {len(self.audio_frames) * 1024 / 16000:.1f}s audio",
            is_registration=(current_tab == 0),
        )

    def clear_recording(self):
        """Clear recorded data"""
        self.recorded_frames = []
        self.audio_frames = []

        current_tab = self.notebook.index(self.notebook.select())
        if current_tab == 0:
            self.reg_frame_count_label.config(text="Frames: 0 | Audio: 0s")
            self.reg_status_label.config(text="Cleared", foreground="green")
            self.reg_register_btn.config(state="disabled")
        else:
            self.auth_frame_count_label.config(text="Frames: 0 | Audio: 0s")
            self.auth_status_label.config(text="Cleared", foreground="green")
            self.auth_authenticate_btn.config(state="disabled")

    def frames_to_base64(self, frames, sample_count=10):
        """Convert frames to base64 strings"""
        if not frames:
            return []
        # Sample frames
        step = max(1, len(frames) // sample_count)
        sampled = frames[::step][:sample_count]

        encoded = []
        for frame in sampled:
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Encode to base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            encoded.append(img_str)

        return encoded

    def audio_to_base64(self):
        """Convert audio frames to base64 WAV"""
        if not self.audio_frames:
            return None

        # Create WAV in memory
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = temp_file.name
        temp_file.close()

        try:
            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(b"".join(self.audio_frames))

            # Read and encode
            with open(temp_path, "rb") as f:
                audio_data = f.read()
                audio_b64 = base64.b64encode(audio_data).decode()
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

        return audio_b64

    def register_user(self):
        """Register user with face and voice"""
        username = self.reg_username.get().strip()
        if not username:
            messagebox.showerror("Error", "Please enter a username")
            return

        if not self.recorded_frames:
            messagebox.showerror("Error", "No video frames recorded")
            return

        if not self.audio_frames:
            messagebox.showerror("Error", "No audio recorded")
            return

        self.log_message(f"Registering user: {username}", is_registration=True)

        # Process in background thread
        thread = threading.Thread(
            target=self._register_user_thread, args=(username,), daemon=True
        )
        thread.start()

    def _register_user_thread(self, username):
        """Background thread for registration"""

        try:
            self.log_message(
                f"Initiating registration for user: {username}",
                is_registration=True,
            )

            # Register face

            self.log_message(
                "Sending face registration request...", is_registration=True
            )

            frames_b64 = self.frames_to_base64(
                self.recorded_frames, sample_count=10
            )

            face_response = requests.post(
                f"{FACE_SERVICE_URL}/register",
                json={"username": username, "frames": frames_b64},
                timeout=30,
            )

            if face_response.status_code == 200:
                result = face_response.json()

                self.log_message(
                    f"✓ Face registration successful: {result.get('faces_detected')} faces from {result.get('total_frames')} frames.",
                    is_registration=True,
                )

            else:
                try:
                    error = face_response.json().get("error", "Unknown error")

                except Exception:
                    error = f"HTTP {face_response.status_code}"

                self.log_message(
                    f"❌ Face registration failed: {error}",
                    is_registration=True,
                )

                messagebox.showerror(
                    "Error", f"Face registration failed: {error}"
                )

                return

            # Register voice

            self.log_message(
                "Sending voice registration request...", is_registration=True
            )

            audio_b64 = self.audio_to_base64()

            voice_response = requests.post(
                f"{VOICE_SERVICE_URL}/register",
                json={"username": username, "audio": audio_b64},
                timeout=30,
            )

            if voice_response.status_code == 200:
                result = voice_response.json()

                self.log_message(
                    f"✓ Voice registration successful: {result.get('message')}",
                    is_registration=True,
                )

            else:
                try:
                    error = voice_response.json().get("error", "Unknown error")

                except Exception:
                    error = f"HTTP {voice_response.status_code}"

                self.log_message(
                    f"❌ Voice registration failed: {error}",
                    is_registration=True,
                )

                messagebox.showerror(
                    "Error", f"Voice registration failed: {error}"
                )

                return

            self.log_message(
                f"✅ Registration complete for {username}!",
                is_registration=True,
            )

            messagebox.showinfo(
                "Success", f"User '{username}' registered successfully!"
            )

        except requests.exceptions.RequestException as e:
            self.log_message(
                f"❌ Connection error during registration: {str(e)}",
                is_registration=True,
            )

            messagebox.showerror(
                "Error", f"Cannot connect to services: {str(e)}"
            )

        except Exception as e:
            self.log_message(
                f"❌ Unexpected error during registration: {str(e)}",
                is_registration=True,
            )

            messagebox.showerror("Error", f"Registration failed: {str(e)}")

    def authenticate_user(self):
        """Authenticate user with face and voice"""

        if not self.recorded_frames:
            messagebox.showerror("Error", "No video frames recorded")

            return

        if not self.audio_frames:
            messagebox.showerror("Error", "No audio recorded")

            return

        face_threshold = float(self.auth_face_threshold.get())

        voice_threshold = float(self.auth_voice_threshold.get())

        self.log_message(
            f"Initiating authentication (Face Threshold: {face_threshold:.2f}, Voice Threshold: {voice_threshold:.2f})...",
            is_registration=False,
        )

        # Process in background thread

        thread = threading.Thread(
            target=self._authenticate_user_thread,
            args=(face_threshold, voice_threshold),
            daemon=True,
        )

        thread.start()

    def _authenticate_user_thread(self, face_threshold, voice_threshold):
        """Background thread for authentication with fusion"""
        try:
            self.log_message(
                "Initiating authentication with Dempster-Shafer fusion...",
                is_registration=False,
            )

            frames_b64 = self.frames_to_base64(
                self.recorded_frames, sample_count=8
            )
            audio_b64 = self.audio_to_base64()

            # Use fusion authentication
            fusion_result = self.auth_handler.authenticate_with_fusion(
                frames_b64, audio_b64, face_threshold, voice_threshold
            )

            # Log errors if any
            for error in fusion_result.get("errors", []):
                self.log_message(f"⚠️ {error}", is_registration=False)

            # Log individual services
            face_ev = fusion_result.get("face_evidence", {})
            voice_ev = fusion_result.get("voice_evidence", {})

            self.log_message(
                f"👤 Face: Score={face_ev.get('score', 0):.3f}, "
                f"Identified={face_ev.get('identified')}",
                is_registration=False,
            )
            self.log_message(
                f"🎤 Voice: Score={voice_ev.get('score', 0):.3f}, "
                f"Identified={voice_ev.get('identified')}",
                is_registration=False,
            )

            # Log fusion result
            self.log_message("=" * 50, is_registration=False)
            self.log_message(
                self.auth_handler.authenticator.get_decision_message(
                    fusion_result
                ),
                is_registration=False,
            )

            # Show detailed report
            report = self.auth_handler.authenticator.get_detailed_report(
                fusion_result
            )
            for line in report.split("\n"):
                self.log_message(line, is_registration=False)

            # Final decision
            if fusion_result["decision"]:
                self.auth_status_label.config(
                    text="Authenticated", foreground="green"
                )
                messagebox.showinfo(
                    "Authenticated",
                    self.auth_handler.authenticator.get_decision_message(
                        fusion_result
                    ),
                )
            else:
                self.auth_status_label.config(
                    text="Authentication Failed", foreground="red"
                )
                messagebox.showwarning(
                    "Authentication Failed", "No matching face or voice found."
                )

            self.auth_authenticate_btn.config(state="disabled")

        except requests.exceptions.RequestException as e:
            self.log_message(
                f"❌ Connection error during authentication: {str(e)}",
                is_registration=False,
            )
            messagebox.showerror(
                "Error", f"Cannot connect to services: {str(e)}"
            )

        except Exception as e:
            self.log_message(
                f"❌ Unexpected error during authentication: {str(e)}",
                is_registration=False,
            )
            messagebox.showerror("Error", f"Authentication failed: {str(e)}")

    def log_message(self, message, is_registration=False):
        """Append timestamped message to appropriate log"""

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        line = f"[{ts}] {message}\n"

        # Append to registration or auth log

        try:
            if is_registration:
                self.reg_log.insert(tk.END, line)

                self.reg_log.see(tk.END)

            else:
                self.auth_log.insert(tk.END, line)

                self.auth_log.see(tk.END)

        except Exception:
            # fallback to printing if UI not ready

            print(line)

    def on_closing(self):
        """Cleanup on app close"""
        # stop recording threads
        self.stop_video = True
        self.is_recording = False
        self.is_recording_audio = False

        # join threads gently
        try:
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=1.0)
        except Exception:
            pass

        # release camera
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

        # terminate pyaudio
        try:
            self.audio.terminate()
        except Exception:
            pass

        # remove any leftover temp files
        for p in self.temp_audio_files:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        self.root.destroy()

    def safe_messagebox(self, kind, title, message):
        """Show thread-safe message boxes from background threads."""
        if kind == "error":
            self.root.after(0, lambda: messagebox.showerror(title, message))
        elif kind == "info":
            self.root.after(0, lambda: messagebox.showinfo(title, message))
        elif kind == "warning":
            self.root.after(0, lambda: messagebox.showwarning(title, message))


if __name__ == "__main__":
    root = tk.Tk()
    app = BiometricAuthUI(root)
    root.mainloop()
