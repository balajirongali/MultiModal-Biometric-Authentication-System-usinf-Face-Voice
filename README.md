A multimodal system to authenticate users using facial and voice traits with Decision-Level Fusion based on Dempster–Shafer Evidence Theory.

📌 Overview

This repository contains the implementation of a Multimodal Biometric Authentication System that combines Face Recognition and Voice Recognition to provide secure and reliable identity verification.

The system employs deep learning–based unimodal authentication and integrates their decisions using Decision-Level Fusion based on Dempster–Shafer (D-S) Evidence Theory, enabling robust authentication under uncertainty and noisy real-world conditions.

This project was developed as part of a Final Year B.Tech Project (2025–26) at S. V. National Institute of Technology (SVNIT), Surat.

🎯 Motivation

Unimodal biometric systems suffer from several limitations, including:

Sensitivity to lighting conditions (Face Recognition)

Background noise and microphone variability (Voice Recognition)

Higher false acceptance and false rejection rates

By combining face and voice modalities, the system improves:

✅ Accuracy

✅ Robustness

✅ Reliability in real-world environments

🏗️ System Architecture

The system consists of four main stages:

1️⃣ Data Acquisition

Camera for capturing face images

Microphone for recording voice samples

2️⃣ Unimodal Authentication

Face Recognition: FaceNet-based facial embeddings

Voice Recognition: SpeechBrain ECAPA-TDNN speaker embeddings

3️⃣ Decision Modeling

Each modality produces an independent authentication decision

4️⃣ Decision-Level Fusion

Decisions are fused using Dempster–Shafer Evidence Theory

A final authentication decision is produced

🧠 Decision-Level Fusion (Dempster–Shafer Theory)

Each biometric modality provides evidence for:

Genuine

Impostor

Evidence is represented using Basic Probability Assignments (BPAs)

Dempster–Shafer theory combines evidence while handling uncertainty and conflicts

Provides greater stability and reliability than simple rule-based fusion methods
