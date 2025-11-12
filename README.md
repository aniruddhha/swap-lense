I cannot directly create or upload files (like a downloadable `.md` file) to your computer.

However, I can provide the complete `README.md` content in a single **markdown code block**. You can easily copy this text and save it as a file named `README.md` on your computer for uploading to GitHub.

Here is the complete content based on your specifications and our troubleshooting:

````markdown
# 🪄 Swap Lense: Live Face Swapping for Apple Silicon (M1/M2)

Swap Lense is a minimal implementation of a live video face-swapping application built using **InsightFace** and **OpenCV**. It is designed to run efficiently on macOS devices with Apple Silicon chips (M1/M2/M3) by leveraging the **CoreML Execution Provider (EP)** for accelerated inference.

## ⚠️ Disclaimer and Ethical Use

This project is intended solely for **educational, experimental, and ethical use**.

The maintainer and contributors of this repository are **not responsible** for any misuse, illegal, or unethical applications of this software, including but not limited to, impersonation, harassment, or creation of deceptive content. Users are solely responsible for adhering to all local laws and ethical guidelines. **Do not use this software for criminal or malicious activities.**

## 🚧 Development Status

This project is currently under **heavy development** and is **not considered production-ready**. Functionality is limited to a simple single-face swap.

## 💻 System Compatibility

| System | Compatibility Status | Notes |
| :--- | :--- | :--- |
| **macOS (Apple Silicon M1/M2/M3)** | **Supported** | Optimized to use the **CoreML Execution Provider (EP)**. |
| **macOS (Intel)** | Untested | Will likely fall back to the slower CPU EP. |
| **Windows, Linux** | Unsupported | Planned for future development. |

---

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.8+**
2.  **Xcode Command Line Tools** (Required for some dependencies on macOS).
3.  **OBS Studio** (Required as a reliable virtual camera intermediary for most applications like WhatsApp, Zoom, and Teams).

### Step 1: Clone the Repository & Setup Environment

```bash
# Clone the repository
git clone [https://github.com/aniruddhha/swap-lense.git](https://github.com/aniruddhha/swap-lense.git)
cd swap-lense

# Create and activate a virtual environment (Recommended)
python3 -m venv venv
source venv/bin/activate

# Install required packages
# NOTE: Ensure you install ONNXRuntime with CoreML support if possible.
pip install -r requirements.txt
````

### Step 2: Download Models

The application requires the InsightFace models.

1.  Create a folder named `models` in the root directory.
2.  Download the **face detector** and **swapper** models:
      * `buffalo_l` (for SCRFD detector and ArcFace landmarks)
      * `inswapper_128.onnx`
3.  Place these files into the newly created `models` folder.

### Step 3: Add Avatars

1.  Create a folder named `avatars` in the root directory.
2.  Add your desired source images and name them sequentially (e.g., `1.jpg`, `2.jpg`, `3.jpg`, etc.). The provided `main.py` code expects files named `1.jpg` through `5.jpg`.

-----

## 🎥 Usage Guide: Running the Swap Lense

### 1\. Launch the Application (CRITICAL STEP)

The correct launch order is **essential** to prevent a black screen issue.

Run the main Python script **first** from your terminal:

```bash
python main.py
```

  * A live preview window will open (this is the OpenCV window).
  * Use keys **1-5** to select an avatar, and **0** to reset (show your real face).
  * The script initializes the virtual camera device. The device name it creates may appear as **"OBS Virtual Camera"** or an empty/generic name in the OBS dropdown.

### 2\. Configure OBS Studio (Essential Intermediary)

OBS is used to create a stable, compatible virtual camera output for communication apps.

1.  **Start the Python script (Step 1).**
2.  Open **OBS Studio**.
3.  In the **Sources** dock, click **+** and select **Video Capture Device**.
4.  In the properties window, set the **Device** dropdown.
      * **Crucial Note:** You must select the device that corresponds to the running Python script (e.g., the device named **"OBS Virtual Camera"** or the empty/generic option that corresponds to the `pyvirtualcam` output).
      * **If successful, your live face-swapped video will immediately appear in the OBS Preview area.**
5.  In the **Controls** dock, click **Start Virtual Camera**.

-----

## 📞 How to Use in Video Conferencing (Meet/Zoom/Teams/WhatsApp)

The key to success is using the **"OBS Virtual Camera"** as the final camera source in your communication application.

### General Steps

1.  **Verify the Python script is running** (Swapped video is visible in the OpenCV window).
2.  **Verify OBS Virtual Camera is running** (Swapped video is visible in the OBS Preview, and the Virtual Camera is **Started**).
3.  In your communication application, go to **Settings** \> **Video/Camera**.
4.  Select **"OBS Virtual Camera"** as your primary camera.

### ⚠️ Troubleshooting (Black Screen)

If you see a black screen in the conferencing app, follow the **Mandatory Restart Sequence**:

1.  **Quit** the video conferencing app (WhatsApp/Zoom/Meet).
2.  **Quit** OBS Studio.
3.  **Quit** the Python script (by pressing `q` in the OpenCV window).
4.  **RESTART** the applications in this exact order: **Python Script** $\rightarrow$ **OBS Studio** (check preview, **Start Virtual Camera**) $\rightarrow$ **Video Conferencing App**.
5.  If using **WhatsApp Desktop**, double-check that both **OBS Studio** and **WhatsApp** have **Screen Recording** permissions enabled in macOS **System Settings** \> **Privacy & Security**.

<!-- end list -->

```

You can copy the code block above and paste it into a new file named `README.md`.
```