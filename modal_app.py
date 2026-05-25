"""Run VisoMaster's face-swap pipeline on a Modal cloud GPU (headless batch).

This deliberately keeps all heavy imports (torch, onnxruntime, app.*) *inside*
the remote functions so the file can be driven from a machine without a GPU or
the CUDA stack installed (e.g. macOS).

Usage
-----
1. One-time: download the ~63 model files into a persistent Volume:

       modal run modal_app.py::download

2. Swap a video (replace every detected face with the source identity):

       modal run modal_app.py --target path/to/video.mp4 \
           --source path/to/source_face.jpg \
           --output out.mp4

   Only replace a specific person (give a reference image of who to replace):

       modal run modal_app.py --target video.mp4 --source new_face.jpg \
           --target-face person_to_replace.jpg --settings settings.json --output out.mp4

3. Many faces -> many people. First extract the distinct faces as references:

       modal run modal_app.py::find_faces --target group.jpg   # -> faces/face_00.png ...

   Then map each one to a source identity in a settings 'swaps' list and run:

       modal run modal_app.py --target group.jpg \
           --settings modal_settings.multiface.example.json --output out.jpg

See modal_settings.example.json (single source) and
modal_settings.multiface.example.json (per-person mapping) for the schema.
"""

import json
from pathlib import Path

import modal

REPO_DIR = Path(__file__).parent
MODELS_DIR = "/models"

# GPU type. "L40S" (48GB) is a safe, fast default. Cheaper: "A10" (24GB) is
# plenty for Inswapper128 + a restorer. Faster/bigger: "A100", "H100".
GPU = "L40S"

# Anything heavier than this likely needs the output streamed via the Volume
# instead of returned inline; fine for typical clips.
FUNCTION_TIMEOUT = 3600  # seconds

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.10"
    )
    # Pip layers first so a later system-deps tweak doesn't re-download torch.
    # Torch built against CUDA 12.4.
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Everything else from PyPI. TensorRT is intentionally omitted — the batch
    # path runs on the CUDA ONNX Runtime provider, and the code already guards
    # `import tensorrt` with try/except.
    .pip_install(
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "scikit-image==0.21.0",
        "pillow==9.5.0",
        "onnx==1.16.1",
        "protobuf==4.23.2",
        "psutil==6.0.0",
        "onnxruntime-gpu==1.20.0",
        "packaging==24.1",
        "PySide6==6.8.2.1",
        "kornia",
        "tqdm",
        "ftfy",
        "regex",
        "numexpr",
        "onnxsim",
        "requests",
        "pyqt-toast-notification==1.3.2",
        "qdarkstyle",
        "pyqtdarktheme",
    )
    # ffmpeg + the system libraries PySide6's Qt6 needs (QtCore/QtGui/QtWidgets
    # are imported by the compute chain even though we run headless/offscreen).
    # binutils provides `strip` for the ABI-tag fix below.
    .apt_install(
        "ffmpeg",
        "binutils",
        "libgl1", "libegl1", "libopengl0", "libglib2.0-0",
        "libdbus-1-3", "libxkbcommon0", "libxkbcommon-x11-0",
        "libfontconfig1", "libfreetype6",
        "libx11-6", "libxext6", "libxrender1", "libsm6", "libice6",
        "libxcb1", "libxcb-cursor0", "libxcb-render0", "libxcb-render-util0",
        "libxcb-shape0", "libxcb-xfixes0", "libxcb-randr0", "libxcb-icccm4",
        "libxcb-image0", "libxcb-keysyms1", "libxcb-shm0", "libxcb-sync1",
        "libxcb-xinerama0", "libxcb-util1", "libxcb-xkb1",
    )
    # Modal runs under gVisor, which reports an old kernel version. PySide6 6.8's
    # Qt libs declare a higher minimum kernel in their .note.ABI-tag, so glibc's
    # loader rejects them ("libQt6Core.so.6: cannot open shared object file").
    # Stripping that note removes the kernel-version gate; the libs run fine.
    .run_commands(
        "find / -path '*/PySide6/*' -name '*.so*' -type f "
        "-exec strip --remove-section=.note.ABI-tag {} + 2>/dev/null; "
        "find / -path '*/shiboken6/*' -name '*.so*' -type f "
        "-exec strip --remove-section=.note.ABI-tag {} + 2>/dev/null; true"
    )
    .env(
        {
            "VISOMASTER_MODELS_DIR": MODELS_DIR,
            # No display in the container; keeps incidental Qt calls headless.
            "QT_QPA_PLATFORM": "offscreen",
        }
    )
    .add_local_dir(
        str(REPO_DIR),
        remote_path="/root/VisoMaster",
        ignore=[
            ".git",
            "**/__pycache__",
            ".github",
            "tensorrt-engines",
            "output",
            "*.mp4",
            "*.avi",
            "*.mkv",
            "*.mov",
        ],
    )
)

app = modal.App("visomaster-swap", image=image)
models_volume = modal.Volume.from_name("visomaster-models", create_if_missing=True)

REMOTE_REPO = "/root/VisoMaster"


def _bootstrap():
    """Make the in-container repo importable and model-dir aware."""
    import os
    import sys

    os.environ.setdefault("VISOMASTER_MODELS_DIR", MODELS_DIR)
    os.chdir(REMOTE_REPO)
    if REMOTE_REPO not in sys.path:
        sys.path.insert(0, REMOTE_REPO)


@app.function(volumes={MODELS_DIR: models_volume}, timeout=FUNCTION_TIMEOUT)
def prepare_models():
    """Populate the Volume: copy the small committed assets, then download the
    model files from the VisoMaster assets release. Safe to re-run (skips
    files already present with a valid hash)."""
    import os
    import shutil

    _bootstrap()

    # Committed assets (plugins, meanshape/lip pkls) baked into the image.
    committed = Path(REMOTE_REPO) / "model_assets"
    if committed.exists():
        shutil.copytree(committed, MODELS_DIR, dirs_exist_ok=True)

    # Import only after the model dir env var is set (paths resolve from it).
    from app.helpers.downloader import download_file
    from app.processors.models_data import models_list

    os.makedirs(MODELS_DIR, exist_ok=True)
    failures = []
    for md in models_list:
        os.makedirs(os.path.dirname(md["local_path"]), exist_ok=True)
        ok = download_file(md["model_name"], md["local_path"], md["hash"], md.get("url"))
        if not ok:
            failures.append(md["model_name"])

    models_volume.commit()
    print(f"\nModel preparation complete. {len(models_list) - len(failures)}/{len(models_list)} ready.")
    if failures:
        print("Failed downloads:", ", ".join(failures))


def _decode_image(buf: bytes):
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode a provided image.")
    return img


@app.function(gpu=GPU, volumes={MODELS_DIR: models_volume}, timeout=FUNCTION_TIMEOUT)
def swap(target_media: bytes, target_filename: str, swaps: list, settings: dict) -> bytes:
    """Swap faces in an image or video and return the encoded result bytes.

    ``swaps`` is a list of mappings; each has ``source_faces`` (list of image
    bytes), an optional ``target_face`` (reference image bytes; None => swap all
    faces), and optional per-face ``parameters``/``swap_model``/``threshold``.
    """
    import tempfile

    _bootstrap()
    from app import headless
    from app.helpers.miscellaneous import is_image_file

    entries = []
    for s in swaps:
        ref = s.get("target_face")
        entry = {
            "target_img": _decode_image(ref) if ref else None,
            "source_imgs": [_decode_image(b) for b in s["source_faces"]],
            "parameters": s.get("parameters") or {},
        }
        if s.get("swap_model"):
            entry["swap_model"] = s["swap_model"]
        if s.get("threshold") is not None:
            entry["threshold"] = s["threshold"]
        entries.append(entry)

    mw = headless.HeadlessMainWindow(device="cuda")
    headless.setup(mw, entries, settings)

    with tempfile.TemporaryDirectory() as tmp:
        suffix = Path(target_filename).suffix or ".mp4"
        in_path = str(Path(tmp) / f"input{suffix}")
        Path(in_path).write_bytes(target_media)

        if is_image_file(target_filename):
            out_path = str(Path(tmp) / "output.png")
            headless.swap_image(mw, in_path, out_path)
        else:
            out_path = str(Path(tmp) / "output.mp4")
            headless.swap_video(mw, in_path, out_path)

        return Path(out_path).read_bytes()


@app.function(gpu=GPU, volumes={MODELS_DIR: models_volume}, timeout=FUNCTION_TIMEOUT)
def detect_faces(target_media: bytes, target_filename: str, frame: int, settings: dict) -> list:
    """Detect distinct faces in an image (or one video frame) and return a list
    of PNG-encoded crops in reading order — references for the ``swaps`` mapping."""
    import tempfile

    import cv2

    _bootstrap()
    from app import headless
    from app.helpers.miscellaneous import is_image_file

    if is_image_file(target_filename):
        img_bgr = _decode_image(target_media)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            vid = str(Path(tmp) / ("v" + (Path(target_filename).suffix or ".mp4")))
            Path(vid).write_bytes(target_media)
            cap = cv2.VideoCapture(vid)
            if frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img_bgr = cap.read()
            cap.release()
            if not ok:
                raise ValueError(f"Could not read frame {frame} from the video.")

    mw = headless.HeadlessMainWindow(device="cuda")
    headless.apply_control_settings(mw, settings or {})

    crops = []
    for face in headless.detect_face_crops(mw, img_bgr):
        ok, buf = cv2.imencode(".png", face["crop"])
        if ok:
            crops.append(bytes(buf))
    return crops


@app.local_entrypoint()
def download():
    """One-time model download into the persistent Volume."""
    prepare_models.remote()


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@app.local_entrypoint()
def find_faces(target: str, out_dir: str = "faces", frame: int = 0, settings: str = ""):
    """Extract each distinct face from the target into numbered crops you can
    pair with source identities in a settings 'swaps' list.

        modal run modal_app.py::find_faces --target group.jpg
    """
    target_path = Path(target)
    if not target_path.is_file():
        raise SystemExit(f"Target media not found: {target}")
    settings_dict = json.loads(Path(settings).read_text()) if settings else {}

    print(f"Detecting faces in {target_path.name} on Modal [{GPU}]...")
    crops = detect_faces.remote(
        target_media=target_path.read_bytes(),
        target_filename=target_path.name,
        frame=frame,
        settings=settings_dict,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, png in enumerate(crops):
        (out / f"face_{i:02d}.png").write_bytes(png)
    print(f"Saved {len(crops)} face(s) to {out}/ (face_00.png ...).")
    print("Pair each with a source in your settings JSON 'swaps' list, then run the swap.")


def _resolve(path: str, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


@app.local_entrypoint()
def main(target: str, source: str = "", source_dir: str = "",
         target_face: str = "", settings: str = "", output: str = ""):
    """Submit a swap job. Runs locally; only file IO happens on this machine.

    Multi-face: put a 'swaps' list in --settings (reference image per person ->
    source identity). Single/all: use --source (+ optional --target-face).
    """
    target_path = Path(target)
    if not target_path.is_file():
        raise SystemExit(f"Target media not found: {target}")
    settings_dict = json.loads(Path(settings).read_text()) if settings else {}

    swaps_payload = []
    swaps_cfg = settings_dict.get("swaps")
    if swaps_cfg:
        base = Path(settings).resolve().parent
        for entry in swaps_cfg:
            srcs = entry.get("source_face")
            srcs = [srcs] if isinstance(srcs, str) else list(srcs or [])
            if not srcs:
                raise SystemExit("Each 'swaps' entry needs a 'source_face'.")
            ref = entry.get("target_face")
            swaps_payload.append({
                "target_face": _resolve(ref, base).read_bytes() if ref else None,
                "source_faces": [_resolve(s, base).read_bytes() for s in srcs],
                "parameters": entry.get("parameters") or {},
                "swap_model": entry.get("swap_model"),
                "threshold": entry.get("threshold"),
            })
        print(f"Multi-face mode: {len(swaps_payload)} mapping(s).")
    else:
        source_paths = []
        if source:
            source_paths.append(Path(source))
        if source_dir:
            source_paths += sorted(
                p for p in Path(source_dir).iterdir()
                if p.suffix.lower() in _IMAGE_SUFFIXES
            )
        if not source_paths:
            raise SystemExit(
                "Provide --source/--source-dir, or a 'swaps' list in --settings."
            )
        swaps_payload.append({
            "target_face": Path(target_face).read_bytes() if target_face else None,
            "source_faces": [p.read_bytes() for p in source_paths],
            "parameters": {},
            "swap_model": None,
            "threshold": None,
        })
        print(f"Single-source mode: {len(source_paths)} source face(s), "
              f"{'one person' if target_face else 'all faces'}.")

    is_image = target_path.suffix.lower() in _IMAGE_SUFFIXES
    out_path = Path(output) if output else target_path.with_name(
        target_path.stem + "_swapped" + (".png" if is_image else ".mp4")
    )

    print(f"Uploading {target_path.name} to Modal [{GPU}]...")
    result = swap.remote(
        target_media=target_path.read_bytes(),
        target_filename=target_path.name,
        swaps=swaps_payload,
        settings=settings_dict,
    )
    out_path.write_bytes(result)
    print(f"Saved swapped output to {out_path}")
