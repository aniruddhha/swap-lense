# main.py
# MVP: Single-face live swap (largest face), InsightFace SCRFD + inswapper_128, 5 avatars (1–5), Reset (0)
# macOS (Apple Silicon) with CoreML EP; falls back to CPU EP if needed.
# Output: Preview window + (optional) OBS Virtual Camera via pyvirtualcam.

import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort  # <-- for provider inspection

# Try to enable CoreML EP first; fall back to CPU
os.environ.setdefault("ORT_LOGGING_LEVEL", "ERROR")
PROVIDERS = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

# Optional: be quiet from OpenCL/Metal logs
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

# ------------------------------
# Config
# ------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H, FPS = 1280, 720, 30
AVATAR_DIR = "avatars"
MODEL_DIR = "models"
AVATAR_PATHS = [os.path.join(AVATAR_DIR, f"{i}.jpg") for i in range(1, 6)]  # 1.jpg ... 5.jpg
SCRFD_NAME = "buffalo_l"  # FaceAnalysis pack (includes SCRFD + ArcFace) – good landmarks/det
SWAPPER_ONNX = os.path.join(MODEL_DIR, "inswapper_128.onnx")

CAROUSEL_HEIGHT = 120     # pixels
CAROUSEL_PAD = 8          # spacing
SHOW_VIRTUALCAM = True    # attempt pyvirtualcam; will auto-disable on failure

# ------------------------------
# Imports that may fail early
# ------------------------------
try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model as get_insight_model
except Exception as e:
    print("Failed to import insightface. pip install insightface", e)
    sys.exit(1)

# Optional virtual cam
vcam = None
try:
    if SHOW_VIRTUALCAM:
        import pyvirtualcam
except Exception:
    SHOW_VIRTUALCAM = False

# ------------------------------
# Utilities
# ------------------------------
def largest_face(faces):
    """Return the face object with largest bbox area or None."""
    if not faces:
        return None
    best = None
    best_area = -1
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = f
    # Ignore tiny faces (<120px high) to avoid background swaps
    if best is not None:
        x1, y1, x2, y2 = best.bbox.astype(int)
        if (y2 - y1) < 120:
            return None
    return best

def draw_carousel(frame_bgr, thumbs_bgr, active_idx):
    """Draw a bottom carousel with 5 thumbnails and an active highlight."""
    h, w = frame_bgr.shape[:2]
    ch = CAROUSEL_HEIGHT
    y0 = h - ch
    overlay = frame_bgr.copy()

    # translucent bg bar
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    frame_bgr = cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0)

    slots = len(thumbs_bgr)
    if slots == 0:
        return frame_bgr
    slot_w = min(int((w - (slots + 1) * CAROUSEL_PAD) / slots), int(ch - 2 * CAROUSEL_PAD))

    for i, t in enumerate(thumbs_bgr):
        if t is None:
            continue
        th = ch - 2 * CAROUSEL_PAD
        tw = slot_w
        thumb = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
        x = CAROUSEL_PAD + i * (tw + CAROUSEL_PAD)
        y = y0 + CAROUSEL_PAD
        frame_bgr[y:y+th, x:x+tw] = thumb
        # label number
        cv2.putText(frame_bgr, str(i+1), (x+6, y+24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        # active highlight
        if active_idx is not None and i == active_idx:
            cv2.rectangle(frame_bgr, (x, y), (x+tw, y+th), (0, 255, 255), 3)

    # Reset hint
    cv2.putText(frame_bgr, "Press 0 to Reset (show real face)",
                (10, y0-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return frame_bgr

def load_image(path):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return img

def prepare_avatar_faces(app, avatar_paths):
    """Detect/align avatar faces once. Returns list of 'source_face' objects or None."""
    results = []
    for p in avatar_paths:
        img = load_image(p)
        if img is None:
            results.append(None)
            continue
        faces = app.get(img)
        if not faces:
            print(f"[avatar] no face found in {p}")
            results.append(None)
            continue
        src = largest_face(faces)
        results.append(src)
    return results

def to_rgb(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

def to_bgr(frame_rgb):
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

# ------------------------------
# Init: camera, models, avatars
# ------------------------------
def main():
    global vcam, SHOW_VIRTUALCAM

    print("Initializing camera…")
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    ok, _ = cap.read()
    if not ok:
        print("ERROR: Could not open camera. Check permissions in System Settings → Privacy → Camera.")
        return

    # ---- Providers info (system level)
    print("ORT available providers:", ort.get_available_providers())

    print("Loading InsightFace (detector+landmarks)…")
    app = FaceAnalysis(name=SCRFD_NAME, providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=(640, 640))  # short-side 640 rule

    # Report the EP used by internal FaceAnalysis models
    try:
        for k, m in getattr(app, "models", {}).items():
            try:
                prov = m.session.get_providers()
            except Exception:
                prov = ["<unknown>"]
            print(f"FaceAnalysis model '{k}' providers:", prov)
    except Exception as e:
        print("Could not enumerate FaceAnalysis model providers:", e)

    print("Loading inswapper model…")
    swapper = get_insight_model(SWAPPER_ONNX, providers=PROVIDERS)
    try:
        print("Swapper providers:", swapper.session.get_providers())
    except Exception as e:
        print("Swapper providers: <unknown>:", e)

    print("Loading avatars…")
    avatar_imgs = [load_image(p) for p in AVATAR_PATHS]
    avatar_faces = prepare_avatar_faces(app, AVATAR_PATHS)

    # Build thumbnails for carousel
    thumbs = []
    for img in avatar_imgs:
        thumbs.append(None if img is None else img.copy())

    # Optional virtual camera
    if SHOW_VIRTUALCAM:
        try:
            vcam = pyvirtualcam.Camera(width=FRAME_W, height=FRAME_H, fps=FPS)
            print(f"Virtual camera started: {vcam.device}")
        except Exception as e:
            print("pyvirtualcam failed; continuing without virtual camera:", e)
            SHOW_VIRTUALCAM = False

    active_idx = None   # None => Reset (no swap)
    last_fps_t = time.time()
    frame_count = 0
    fps_smoothed = 0.0

    print("Running. Window keys: 1–5 select avatar, 0 reset, q quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Detect faces each frame (simple MVP). If slow, do every 2–3 frames and cache.
        faces = app.get(frame)
        tgt = largest_face(faces)

        if tgt is not None and active_idx is not None and avatar_faces[active_idx] is not None:
            try:
                # paste_back=True returns the full composited frame (with internal mask & blending)
                frame = swapper.get(frame, tgt, avatar_faces[active_idx], paste_back=True)
            except Exception:
                pass

        # HUD: status + FPS + EP label
        frame_count += 1
        t_now = time.time()
        if t_now - last_fps_t >= 0.5:
            fps_inst = frame_count / (t_now - last_fps_t)
            fps_smoothed = 0.6 * fps_smoothed + 0.4 * fps_inst if fps_smoothed > 0 else fps_inst
            frame_count = 0
            last_fps_t = t_now

        status = "OFF" if active_idx is None else f"Avatar {active_idx+1}"
        cv2.putText(frame, f"Swap: {status}  FPS: {fps_smoothed:.1f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 255, 10), 2, cv2.LINE_AA)

        try:
            ep_label = (swapper.session.get_providers() or ["CPU"])[0]
        except Exception:
            ep_label = "CPU"
        cv2.putText(frame, f"EP:{ep_label}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 255), 2, cv2.LINE_AA)

        # Draw bottom carousel
        frame = draw_carousel(frame, thumbs, active_idx)

        # Show preview
        cv2.imshow("FaceSwap MVP (Press 1-5, 0=Reset, q=Quit)", frame)

        # Send to virtual cam if available
        if SHOW_VIRTUALCAM and vcam is not None:
            vcam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            vcam.sleep_until_next_frame()

        # Key handling (OpenCV)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            idx = int(chr(key)) - 1
            if avatar_faces[idx] is not None:
                active_idx = idx
                print(f"Activated avatar {idx+1}")
            else:
                print(f"Avatar {idx+1} not ready (no face found or image missing).")
        elif key == ord('0'):
            active_idx = None
            print("Reset: showing real face.")

    cap.release()
    cv2.destroyAllWindows()
    if vcam is not None:
        try:
            vcam.close()
        except Exception:
            pass

if __name__ == "__main__":
    # Basic model existence check
    missing = []
    if not os.path.exists(SWAPPER_ONNX):
        missing.append(SWAPPER_ONNX)
    if missing:
        print("Missing model files:\n  " + "\n  ".join(missing))
        print("Download InsightFace ONNX models and place them under ./models/")
        sys.exit(1)
    main()
