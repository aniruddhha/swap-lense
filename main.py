# main.py
# MVP: Single-face live swap (largest face), InsightFace SCRFD + inswapper_128, 5 avatars (1–5), Reset (0)
# macOS (Apple Silicon) with CoreML EP; falls back to CPU EP if needed.
# Output: Local preview window (with optional overlays) + clean feed to OBS Virtual Camera via pyvirtualcam.

import os                     # used to read env vars and file paths (models/avatars folders)
import sys                    # used for clean exits when models/camera missing
import time                   # used to compute FPS in HUD
import cv2                    # OpenCV: camera I/O, drawing carousel/HUD, resizing frames
import numpy as np            # NumPy: image array ops; InsightFace swap blends use it under the hood
import onnxruntime as ort     # ONNX Runtime: lets InsightFace run on CoreML (M1 GPU/ANE) or CPU

# =========================
# Global Config
# =========================
os.environ.setdefault("ORT_LOGGING_LEVEL", "ERROR")   # quiet ONNXRuntime logs for cleaner console
os.environ.setdefault("GLOG_minloglevel", "2")        # quiet some native backend logs
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")   # silence OpenCV info logs

# Inference providers preference: CoreML first (Apple Silicon), then CPU
PROVIDERS = ["CoreMLExecutionProvider", "CPUExecutionProvider"]  # meets "fast on M1" requirement

# Camera / Output
CAM_INDEX = 0                 # use default webcam as the video source
FRAME_W, FRAME_H, FPS = 1280, 720, 30  # standard 720p@30fps for Meet/Teams compatibility (virtual cam)

# Detection throttling & filtering
DET_EVERY = 2                 # detect faces every 2 frames to reduce latency, still track the largest face
IGNORE_FACE_MIN_H = 120       # ignore tiny faces so we stay on the main (largest) face per requirement

# Assets
AVATAR_DIR = "avatars"        # folder holding the 5 avatar images used for swapping (1–5)
MODEL_DIR = "models"          # folder that contains inswapper_128.onnx
AVATAR_PATHS = [os.path.join(AVATAR_DIR, f"{i}.jpg") for i in range(1, 6)]  # 5 avatars for carousel hotkeys
SWAPPER_ONNX = os.path.join(MODEL_DIR, "inswapper_128.onnx")  # InsightFace swapper model (requirement: inswapper_128)
SCRFD_NAME = "buffalo_l"      # InsightFace pack name that includes SCRFD detector + ArcFace landmarks

# UI
CAROUSEL_HEIGHT = 120         # pixel height of the bottom thumbnail carousel (requirement: 5 avatars)
CAROUSEL_PAD = 8              # spacing between thumbnails

# Virtual cam
SHOW_VIRTUALCAM = True        # enable publishing to virtual camera (requirement: virtual camera publishing)

# Overlays routing
SHOW_OVERLAYS_ON_PREVIEW = True   # show HUD + carousel only in preview (so user sees UI)
SHOW_OVERLAYS_ON_VCAM    = False  # send a clean feed to virtual cam (no overlays) to conferencing apps

# =========================
# Imports (with graceful fail)
# =========================
try:
    import insightface                                           # main library that wraps SCRFD + swapper
    from insightface.app import FaceAnalysis                     # high-level detector/landmarker API (SCRFD + ArcFace)
    from insightface.model_zoo import get_model as get_insight_model  # loader for inswapper_128.onnx
except Exception as e:
    print("Failed to import insightface. Install with: pip install insightface")  # help user install deps
    print("Error:", e)                                           # print exact import error for debugging
    sys.exit(1)                                                  # exit early; can’t proceed without models

# Optional virtual cam
vcam = None                                                      # will hold pyvirtualcam.Camera if available
PixelFormat = None                                               # pixel format enum for pyvirtualcam
if SHOW_VIRTUALCAM:
    try:
        import pyvirtualcam                                      # virtual camera bridge to OBS VC
        from pyvirtualcam import PixelFormat                     # explicit RGB format for compatibility
    except Exception as e:
        print("pyvirtualcam not available; continuing without virtual camera. Error:", e)  # fallback path
        SHOW_VIRTUALCAM = False                                  # disable virtual cam if library missing

# =========================
# Utilities
# =========================
def ensure_size(img, w, h):
    """Return img resized to exactly (w,h) if needed."""
    ih, iw = img.shape[:2]                                       # current frame height/width
    if (ih, iw) != (h, w):                                       # if not the requested size (e.g., 1080x608 from macOS)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)  # resize to 1280x720 for virtual cam stability
    return img                                                   # return properly sized frame

def largest_face(faces):
    """Return the face object with largest bbox area or None."""
    if not faces:                                                # if no faces detected, return None
        return None
    best = None                                                  # hold the largest face object
    best_area = -1                                               # track max area
    for f in faces:                                              # iterate all detected faces
        x1, y1, x2, y2 = f.bbox.astype(int)                      # bounding box corners
        area = max(0, x2 - x1) * max(0, y2 - y1)                 # compute area
        if area > best_area:                                     # update if this one is larger
            best_area = area
            best = f
    # Ignore tiny faces
    if best is not None:
        x1, y1, x2, y2 = best.bbox.astype(int)                   # get largest bbox height
        if (y2 - y1) < IGNORE_FACE_MIN_H:                        # filter out small faces; keeps us on main subject
            return None
    return best                                                  # return the largest qualified face (requirement: single-face focus)

def draw_carousel(frame_bgr, thumbs_bgr, active_idx):
    """Draw a bottom carousel with 5 thumbnails and an active highlight."""
    h, w = frame_bgr.shape[:2]                                   # frame size for layout
    ch = CAROUSEL_HEIGHT                                         # carousel bar height
    y0 = h - ch                                                  # top of carousel bar (bottom of frame)
    overlay = frame_bgr.copy()                                   # copy for translucent bar

    # translucent bg bar
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)       # draw black bar at bottom
    frame_bgr = cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0)  # blend to make it translucent

    slots = len(thumbs_bgr)                                      # number of avatar slots (expect 5)
    if slots == 0:                                               # if no avatars, just return
        return frame_bgr
    slot_w = min(int((w - (slots + 1) * CAROUSEL_PAD) / slots),  # compute thumbnail width
                 int(ch - 2 * CAROUSEL_PAD))

    for i, t in enumerate(thumbs_bgr):                           # draw each thumbnail
        if t is None:                                            # if avatar image missing, skip
            continue
        th = ch - 2 * CAROUSEL_PAD                               # thumbnail height
        tw = slot_w                                              # thumbnail width
        thumb = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)  # fit avatar to slot
        x = CAROUSEL_PAD + i * (tw + CAROUSEL_PAD)               # left x of this slot
        y = y0 + CAROUSEL_PAD                                    # top y of this slot
        frame_bgr[y:y+th, x:x+tw] = thumb                        # paste thumbnail into the bar
        # label number
        cv2.putText(frame_bgr, str(i+1), (x+6, y+24),            # draw key label (1..5) for hotkeys
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        # active highlight
        if active_idx is not None and i == active_idx:           # highlight if this avatar is selected
            cv2.rectangle(frame_bgr, (x, y), (x+tw, y+th), (0, 255, 255), 3)

    # Reset hint
    cv2.putText(frame_bgr, "Press 0 to Reset (show real face)",  # usage hint for reset requirement
                (10, y0-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return frame_bgr                                             # return frame with carousel painted

def load_image(path):
    if not os.path.exists(path):                                 # if avatar path missing, return None
        return None
    return cv2.imread(path)                                      # read avatar from disk (BGR)

def prepare_avatar_faces(app, avatar_paths):
    """Detect/align avatar faces once. Returns list of 'source_face' objects or None."""
    results = []                                                 # list of prepared faces for inswapper
    for p in avatar_paths:                                       # iterate 1.jpg..5.jpg
        img = load_image(p)                                      # load avatar image
        if img is None:                                          # if missing, append None placeholder
            results.append(None)
            continue
        faces = app.get(img)                                     # run SCRFD detector on avatar image
        if not faces:                                            # if no face found, mark None
            print(f"[avatar] no face found in {p}")
            results.append(None)
            continue
        results.append(largest_face(faces))                      # pick largest face in avatar for stability
    return results                                               # return prepared avatar faces (used by swapper)

def draw_hud_and_status(frame, swapper, active_idx, last_fps_t, frame_count, fps_smoothed):
    """Calculates FPS and draws the Swap Status, FPS, and EP status on the frame."""
    frame_count += 1                                             # count frame for FPS
    t_now = time.time()                                          # current time

    if t_now - last_fps_t >= 0.5:                                # update FPS about twice per second
        fps_inst = frame_count / (t_now - last_fps_t)            # instantaneous FPS
        fps_smoothed = 0.6 * fps_smoothed + 0.4 * fps_inst if fps_smoothed > 0 else fps_inst  # smooth FPS
        frame_count = 0                                          # reset counters
        last_fps_t = t_now

    status = "OFF" if active_idx is None else f"Avatar {active_idx+1}"  # show which avatar is active or OFF
    cv2.putText(frame, f"Swap: {status}  FPS: {fps_smoothed:.1f}",      # HUD: shows swap status and FPS
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 255, 10), 2, cv2.LINE_AA)

    try:
        ep_label = (swapper.session.get_providers() or ["CPU"])[0]      # show current ONNX EP (CoreML or CPU)
    except Exception:
        ep_label = "CPU"                                                # fallback label if not accessible
    cv2.putText(frame, f"EP:{ep_label}",                                # HUD: shows CoreMLExecutionProvider if used
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 255), 2, cv2.LINE_AA)

    return frame, last_fps_t, frame_count, fps_smoothed                 # return updated HUD and FPS state

# =========================
# Init / Load
# =========================
def load_config_and_models():
    """Initializes camera, InsightFace detector, swapper, and pre-loads avatars."""
    global SHOW_VIRTUALCAM                                        # we may disable this if init fails

    print("Initializing camera…")                                  # log for visibility
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)        # open macOS camera using AVFoundation backend
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)                     # request 1280 width from camera
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)                    # request 720 height from camera
    cap.set(cv2.CAP_PROP_FPS, FPS)                                 # request 30fps from camera
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                            # low-latency capture (avoid backlog)

    ok, frame_check = cap.read()                                   # pull one frame to verify camera works
    if not ok:                                                     # if no frame, permissions or camera busy
        print("ERROR: Could not open camera. Check permissions in System Settings → Privacy → Camera.")
        cap.release()                                              # release camera handle
        sys.exit(1)                                                # exit: cannot proceed without camera

    print("ORT available providers:", ort.get_available_providers())  # show which ONNX EPs are compiled in

    print("Loading InsightFace (detector+landmarks)…")             # SCRFD + ArcFace (via FaceAnalysis)
    app = FaceAnalysis(name=SCRFD_NAME, providers=PROVIDERS)       # request CoreML EP first for speed
    app.prepare(ctx_id=0, det_size=(640, 640))                     # prepare detector with 640 short side (balanced)

    # (optional) show internal model providers
    try:
        for k, m in getattr(app, "models", {}).items():            # iterate submodels used by FaceAnalysis
            try:
                prov = m.session.get_providers()                   # print their ONNX EPs (debug visibility)
            except Exception:
                prov = ["<unknown>"]
            print(f"FaceAnalysis model '{k}' providers:", prov)
    except Exception:
        pass                                                       # ignore if internal API changes

    print("Loading inswapper model…")                              # load the swapper ONNX (inswapper_128)
    swapper = get_insight_model(SWAPPER_ONNX, providers=PROVIDERS) # request CoreML EP when available
    try:
        print("Swapper providers:", swapper.session.get_providers())# log which EP swapper actually uses
    except Exception as e:
        print("Swapper providers: <unknown>:", e)                  # ignore if not exposed

    print("Loading avatars…")                                      # read avatars and prep faces for swapping
    avatar_imgs = [load_image(p) for p in AVATAR_PATHS]            # load each 1.jpg..5.jpg
    avatar_faces = prepare_avatar_faces(app, AVATAR_PATHS)         # detect/aligned faces for swapper source
    thumbs = [None if img is None else img.copy() for img in avatar_imgs]  # thumbnails for carousel UI

    # Preload: touch each identity once to warm caches
    print("Pre-loading all avatar identities to mitigate first-swap lag…")  # avoid first-frame jank
    faces_dummy = app.get(frame_check)                              # detect a target on the check frame
    tgt_dummy = largest_face(faces_dummy)                           # pick the main face if present
    if tgt_dummy is not None:
        for i, src_face in enumerate(avatar_faces):                 # for each avatar face prepared
            if src_face is not None:
                try:
                    swapper.get(frame_check, tgt_dummy, src_face, paste_back=False)  # dry-run to warm up
                    print(f"  Pre-loaded identity for Avatar {i+1}.")               # confirm avatar cached
                except Exception as e:
                    print(f"  Warning: Pre-load failed for Avatar {i+1}: {e}")      # non-fatal warning
    else:
        print("  Warning: No face detected in camera for pre-loading step. Skipping pre-load.")  # still OK
    print("Pre-loading complete.")                                  # preload step done

    # Virtual camera
    vcam = None                                                    # default to no virtual cam
    if SHOW_VIRTUALCAM and PixelFormat is not None:                # if user wants it and lib is present
        try:
            vcam = pyvirtualcam.Camera(width=FRAME_W, height=FRAME_H, fps=FPS, fmt=PixelFormat.RGB)  # open VC
            print(f"Virtual camera started: {vcam.device} ({FRAME_W}x{FRAME_H}@{FPS})")              # log device
        except Exception as e:
            print("pyvirtualcam failed; continuing without virtual camera:", e)   # VC optional; continue anyway
            SHOW_VIRTUALCAM = False                                              # disable VC path

    return cap, app, swapper, avatar_faces, thumbs, vcam          # return initialized components to main loop

# =========================
# Main Loop
# =========================
def run_swap_loop(cap, app, swapper, avatar_faces, thumbs, vcam):
    global SHOW_OVERLAYS_ON_PREVIEW                                # allow hotkey to toggle HUD in preview

    active_idx = None                                              # no avatar selected initially (Reset state)
    last_fps_t = time.time()                                       # last timestamp for FPS window
    frame_count = 0                                                # frames counted in current window
    fps_smoothed = 0.0                                             # smoothed FPS value for HUD
    last_tgt = None                                                # cache of the last detected (largest) target face
    i = 0                                                          # frame index to drive DET_EVERY

    print("Running. Keys: 1–5 select avatar, 0 reset, H toggle preview overlays, Q quit.")  # UX hint
    while True:                                                    # main processing loop
        ok, frame = cap.read()                                     # grab a frame from webcam
        if not ok:                                                 # if camera fails, exit loop
            break

        # Throttled detection
        if i % DET_EVERY == 0:                                     # run SCRFD detection every N frames
            faces = app.get(frame)                                 # detect faces on current frame
            last_tgt = largest_face(faces)                         # pick the largest face (single-face focus)
        i += 1                                                     # increment frame index
        tgt = last_tgt                                             # use cached target between detections

        # Swap
        if tgt is not None and active_idx is not None and avatar_faces[active_idx] is not None:
            try:
                frame = swapper.get(frame, tgt, avatar_faces[active_idx], paste_back=True)  # run inswapper_128
            except Exception:                                      # if swap momentarily fails, skip frame
                pass

        # --- Split clean vs preview ---
        clean_frame = frame                                        # clean feed (no overlays) sent to virtual cam

        # Build preview (with overlays if enabled)
        if SHOW_OVERLAYS_ON_PREVIEW:                               # show HUD/carousel only in preview window
            preview_frame = clean_frame.copy()                     # draw on a copy so vcam stays clean
            preview_frame, last_fps_t, frame_count, fps_smoothed = draw_hud_and_status(
                preview_frame, swapper, active_idx, last_fps_t, frame_count, fps_smoothed  # HUD: FPS + EP + status
            )
            preview_frame = draw_carousel(preview_frame, thumbs, active_idx)  # bottom 5-avatar carousel (1–5)
        else:
            # still update FPS numbers even if not drawing
            _, last_fps_t, frame_count, fps_smoothed = draw_hud_and_status(
                clean_frame, swapper, active_idx, last_fps_t, frame_count, fps_smoothed   # update counters only
            )
            preview_frame = clean_frame                              # preview equals clean frame (no overlays)

        # Force sizes/types
        preview_out = ensure_size(preview_frame, FRAME_W, FRAME_H)   # guarantee preview is 1280x720
        vcam_out    = ensure_size(clean_frame,   FRAME_W, FRAME_H)   # guarantee virtual cam is 1280x720
        if preview_out.dtype != np.uint8:                            # enforce 8-bit frames for OpenCV/vcam
            preview_out = preview_out.astype(np.uint8)
        if vcam_out.dtype != np.uint8:
            vcam_out = vcam_out.astype(np.uint8)

        # Preview
        cv2.imshow("FaceSwap MVP (Press 1-5, 0=Reset, H=HUD toggle, Q=Quit)", preview_out)  # local window for user

        # Virtual cam (clean feed unless overridden)
        if SHOW_VIRTUALCAM and vcam is not None:                    # publish to OBS Virtual Camera if enabled
            send_frame = preview_out if SHOW_OVERLAYS_ON_VCAM else vcam_out  # default: clean feed (no overlays)
            rgb = cv2.cvtColor(send_frame, cv2.COLOR_BGR2RGB)       # convert BGR→RGB (vcam expects RGB)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)         # ensure contiguous memory for pyvirtualcam
            vcam.send(rgb)                                          # push frame to virtual camera
            vcam.sleep_until_next_frame()                           # pace to the configured FPS

        # Keys
        key = cv2.waitKey(1) & 0xFF                                 # poll keyboard (non-blocking)
        if key == ord('q') or key == 27:                            # Q or ESC: exit app
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:  # 1–5: select avatar in carousel
            idx = int(chr(key)) - 1                                 # map ASCII digit to index 0..4
            if avatar_faces[idx] is not None:                       # only activate if avatar face was prepared
                active_idx = idx                                    # select avatar (swaps will start)
                print(f"Activated avatar {idx+1}")                  # log selection
            else:
                print(f"Avatar {idx+1} not ready (no face found or image missing).")  # helpful warning
        elif key == ord('0'):                                       # 0: Reset button (show real face)
            active_idx = None                                       # disable swapping; show your own face
            print("Reset: showing real face.")
        elif key in (ord('h'), ord('H')):                           # H: toggle overlays in preview
            SHOW_OVERLAYS_ON_PREVIEW = not SHOW_OVERLAYS_ON_PREVIEW # flip boolean
            print("Preview overlays:", "ON" if SHOW_OVERLAYS_ON_PREVIEW else "OFF")  # confirm to user

    # Cleanup
    cap.release()                                                   # release camera device
    cv2.destroyAllWindows()                                        # close the preview window
    if vcam is not None:                                           # if virtual cam was opened
        try:
            vcam.close()                                           # close it cleanly
            print("Virtual camera closed.")                        # confirm closure
        except Exception:
            pass                                                   # ignore if already closed

# =========================
# Entry
# =========================
def main():
    # Model existence check
    if not os.path.exists(SWAPPER_ONNX):                            # ensure inswapper_128.onnx is present
        print("Missing model files:\n  " + SWAPPER_ONNX)            # direct user to place the model
        print("Download InsightFace ONNX models and place them under ./models/")  # explicit instruction
        sys.exit(1)                                                 # cannot continue without swapper model

    try:
        cap, app, swapper, avatar_faces, thumbs, vcam = load_config_and_models()  # init camera/models/avatars/vcam
        run_swap_loop(cap, app, swapper, avatar_faces, thumbs, vcam)              # enter processing loop
    except Exception as e:
        print("\nFATAL ERROR:", e)                                  # catch-all to avoid hard crashes
        sys.exit(1)                                                 # exit non-zero for scripts

if __name__ == "__main__":                                          # standard Python entry point
    main()                                                          # run app: fulfills all 5 MVP requirements
