# 🪄 Swap Lense: Live Face Swapping for Apple Silicon (M1/M2/M3)

Swap Lense is a minimal live **face-swapping** app built with **InsightFace** + **OpenCV**, optimized for **Apple Silicon** using **ONNX Runtime (Core ML Execution Provider)**. It runs locally on your Mac (no cloud), aims for privacy-first calls, and can publish the video via a **Virtual Camera** for Meet/Teams/Zoom.

---

## ⚠️ Disclaimer & Ethical Use

This software is for **educational and ethical** use only. Do **not** use it for impersonation, harassment, or deceptive content. You are solely responsible for following laws, platform policies, and consent requirements. Prefer **AI‑generated avatars** over real-person likenesses (especially celebrities).

---

## ✅ Compatibility

| System | Status | Notes |
|---|---|---|
| macOS (Apple Silicon M1/M2/M3) | **Supported** | Uses **Core ML EP** when available; falls back to CPU EP. |
| macOS (Intel) | Untested | Likely CPU-only; performance may be limited. |
| Windows / Linux | Not yet | Planned later. |

---

## 🚀 Quick Start

### 1) Clone & Virtual Environment
```bash
git clone https://github.com/aniruddhha/swap-lense.git
cd swap-lense

python -m venv .venv
source .venv/bin/activate
python --version  
```

### 2) Install Requirements
```bash
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```
> If you hit `onnxruntime-coreml` issues, stick to **onnxruntime==1.23.2** and **skip** the coreml extra; Core ML EP is bundled in main `onnxruntime` for arm64 wheels on recent versions. Performance will still be solid.

### 3) Model Files
Create a `models/` folder and place:
- `inswapper_128.onnx` (face swapper)
- *Detector/landmarks* come from InsightFace `buffalo_l` pack automatically (downloaded on first run).

```
swap-lense/
├── main.py
├── requirements.txt
├── models/
│   └── inswapper_128.onnx
└── avatars/
    ├── 1.jpg
    ├── 2.jpg
    ├── 3.jpg
    ├── 4.jpg
    └── 5.jpg
```

### 4) Avatars
Add 5 images into `avatars/` as `1.jpg … 5.jpg`. Prefer **front-facing, evenly lit** images. For safety, use **AI-generated** faces.

---

## ▶️ Run
```bash
python main.py
```
- **1–5**: pick avatar
- **0**: reset (show your real face)
- **q**: quit

The app shows a live preview window and (optionally) publishes to a Virtual Camera if available.

---

## 🎥 Virtual Camera (to use in Meet/Teams/Zoom)

### Option A — OBS Virtual Camera (recommended and most compatible)
1. **Install OBS from the official .dmg (Apple Silicon build).**
2. Ensure the plugin exists (after install or manual copy):
   - `/Library/CoreMediaIO/Plug-Ins/DAL/obs-mac-virtualcam.plugin`
3. Open **OBS** and click **Start Virtual Camera** (canvas may stay black; that’s normal).
4. Run this app; console prints: `Virtual camera started: OBS Virtual Camera`.
5. Open **QuickTime → New Movie Recording → ⌄ → Camera: OBS Virtual Camera** — you should see the swap feed.
6. In **Meet/Teams/Zoom**, go to **Settings → Video/Camera → select “OBS Virtual Camera.”**

> If OBS Virtual Camera is missing after dmg install, manually copy the plugin from inside the app bundle:
> - From: `/Applications/OBS.app/Contents/Resources/obs-mac-virtualcam.plugin`
> - To: `/Library/CoreMediaIO/Plug-Ins/DAL/`
> Then log out/in and start OBS Virtual Camera.

### macOS Permissions & Gotchas
- System Settings → **Privacy & Security → Camera**: allow **OBS**, your **browser**, **Teams**.
- Only **one app** can hold the **physical** camera. Close FaceTime/Photo Booth/Zoom if they’re open.
- In Chrome, try toggling **Hardware Acceleration** if you see black (then relaunch).

---

## 🧰 Troubleshooting

**Black screen in apps but preview works**
- Ensure OBS Virtual Camera is **Started**.
- Verify QuickTime sees **OBS Virtual Camera**.
- Close and reopen the conferencing app **after** starting OBS VC.
- Make sure our app is not minimized or blocked by permissions.

**OBS Virtual Camera not installed**
- Copy the plugin:
  ```bash
  sudo mkdir -p /Library/CoreMediaIO/Plug-Ins/DAL
  sudo cp -R "/Applications/OBS.app/Contents/Resources/obs-mac-virtualcam.plugin" \
             /Library/CoreMediaIO/Plug-Ins/DAL/
  # log out/in (or reboot) after copying
  ```

**Shape mismatch error (pyvirtualcam)**
- We force-resize to **1280×720** before sending; if you change `FRAME_W/H`, keep them consistent.

**Video looks fast-forward**
- Use the updated script that records **after warm-up** and sets the MP4 FPS to the **measured** FPS.

---

## 🧪 Feature Scope (MVP)

- Single-face focus (largest face box)
- **SCRFD + inswapper_128** with simple face parsing mask (built-in)
- 5 avatars in a bottom carousel (hotkeys **1–5**)
- **Reset** button (**0**) to show your real face
- Optional **virtual camera** publishing
- Optional **MP4 recording** with correct FPS pacing

---

## 🔐 Privacy Notes

- All processing is **local** on your Mac.
- No images or video leave your device.
- Prefer **AI-generated avatars** to avoid likeness rights issues.
- If you demo with celebrity images, state clearly it’s for **testing only** (and remove them).

---

## 📄 License

MIT (or your preferred permissive license). Use responsibly.

---

## 💬 Credits

- [InsightFace](https://github.com/deepinsight/insightface)
- [OpenCV](https://opencv.org/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [OBS Studio](https://obsproject.com/)

