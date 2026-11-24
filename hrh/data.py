"""THINGS-EEG2 data access (official OSF routes; verified file GUIDs).

Gifford, Dwivedi, Roig & Cichy (2022), "A large and rich EEG dataset for modeling
human visual object recognition", NeuroImage. Project: osf.io/3jk45.
Ordering of the 200 test images follows image_metadata.npy — this is the only
place stimulus order is determined; a silent misalignment corrupts every RDM.
"""
from __future__ import annotations

import glob
import os
import subprocess

import numpy as np

CONFIG = {
    # Preprocessed EEG, THINGS-EEG2 (10 subjects). Official project: osf.io/3jk45
    # Verified download route (from the EEG-Image-Reconstruction README):
    #   wget "https://files.de-1.osf.io/v1/resources/anp5v/providers/osfstorage/?zip=" -O thingseeg2_preproc.zip
    #   -> contains sub-01.zip .. sub-10.zip, each with:
    #        preprocessed_eeg_training.npy, preprocessed_eeg_test.npy
    #   The .npy files are DICTS with keys:
    #     'preprocessed_eeg_data'  (n_images, n_reps, 63, n_times)
    #     'ch_names', 'times'
    #   TEST set = 200 images (each ~80 repetitions). Train = 16,540 images.
    # Per-file direct URLs (verified via the OSF API, exact sizes):
    #   Preprocessed EEG (anp5v): 10 x sub-XX.zip, 1.07 GB each (test+train npy inside)
    #   Image set (y63gw): test_images.zip = 10 MB, training = 0.66 GB, image_metadata.npy tiny
    "SUBJECT_ZIP_URL": "https://files.de-1.osf.io/v1/resources/anp5v/providers/osfstorage/",
    # verified via the OSF API — one GUID per subject's zip
    "SUBJECT_ZIPS": {
        "sub-01": "62f6b4ff170d520c73c27c1e", "sub-02": "62f6b6eb170d520c73c27e35",
        "sub-03": "62f6b7060beb5f05deb0c554", "sub-04": "62f6b70721d7ec0d04c71205",
        "sub-05": "62f6b7085775130b6bf252d5", "sub-06": "62f6bfd521d7ec0d50c70d92",
        "sub-07": "62f6bfe30beb5f05f3b0a34a", "sub-08": "62f6bfe45775130baaf2516e",
        "sub-09": "62f6bfe60beb5f05efb0a877", "sub-10": "62f6bfe52331f10c51b0ec10",
    },
    "TEST_IMAGES_ZIP_URL": ("https://files.de-1.osf.io/v1/resources/y63gw/providers/osfstorage/"
                            "60d8373df83de2006bb02b23"),  # test_images.zip (10 MB)
    "IMAGE_METADATA_URL": ("https://files.de-1.osf.io/v1/resources/y63gw/providers/osfstorage/"
                           "625c3f0d0bfe3c21cf182889"),   # image_metadata.npy
    "TEST_FILE": "preprocessed_eeg_test.npy",
    "N_TEST_IMAGES": 200,
    "IMG_DIR": "data/things/test_images",
    # image_metadata.npy (from the same Image_set component) maps the EEG test
    # image index 1..200 -> THINGS image id/filename. If absent we fall back to
    # sorted filenames — VERIFY with --check that ordering matches.
    "IMAGE_METADATA_NPY": "data/things/image_metadata.npy",
}


def _remote_size(url: str) -> int:
    import urllib.request
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def fetch(url: str, path: str, tries: int = 50, chunk: int = 1 << 20) -> str:
    """Resumable download (HTTP Range) with retries; verifies the final size.
    OSF regularly drops long transfers mid-stream; this resumes where it stopped."""
    import time
    import urllib.request
    size = _remote_size(url)
    for attempt in range(tries):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if size and have >= size:
            break
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "ab" if have else "wb") as f:
                if have and r.status != 206:            # server ignored Range: restart
                    f.truncate(0)
                    have = 0
                last = -1
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    have += len(buf)
                    pct = int(100 * have / size) if size else 0
                    if pct // 10 != last:
                        last = pct // 10
                        print(f"  {os.path.basename(path)}: {pct}% of {size / 1e9:.2f} GB",
                              flush=True)
        except Exception as exc:                        # network hiccup: back off, resume
            print(f"  retry {attempt + 1}/{tries} after: {exc}", flush=True)
            time.sleep(min(5 * (attempt + 1), 60))
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if size and have != size:
        raise RuntimeError(f"{path}: got {have} of {size} bytes after {tries} attempts")
    return path


def download_subject(subject: str, eeg_dir: str, keep_zip: bool = False) -> str:
    """Download one subject's 1.07 GB zip (resumable), extract only the test-set
    file, and delete the zip unless keep_zip."""
    os.makedirs(os.path.join(eeg_dir, subject), exist_ok=True)
    dest = os.path.join(eeg_dir, subject, CONFIG["TEST_FILE"])
    if os.path.exists(dest):
        return dest
    zip_path = os.path.join(eeg_dir, f"{subject}.zip")
    fetch(CONFIG["SUBJECT_ZIP_URL"] + CONFIG["SUBJECT_ZIPS"][subject], zip_path)
    subprocess.run(["unzip", "-o", "-q", zip_path, f"{subject}/{CONFIG['TEST_FILE']}",
                    "-d", eeg_dir], check=True)
    if not os.path.exists(dest):
        raise FileNotFoundError(f"{dest} missing after extraction")
    if not keep_zip:
        os.remove(zip_path)
    return dest


def ensure_images(img_dir: str, n: int) -> None:
    """Fetch test_images.zip (10 MB) + image_metadata.npy if not present."""
    os.makedirs(img_dir, exist_ok=True)
    meta_path = CONFIG["IMAGE_METADATA_NPY"]
    if not os.path.exists(meta_path):
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        subprocess.run(["wget", "-q", CONFIG["IMAGE_METADATA_URL"], "-O", meta_path], check=True)
    if len(glob.glob(os.path.join(img_dir, "*.*"))) < n:
        subprocess.run(["wget", "-q", "--show-progress", CONFIG["TEST_IMAGES_ZIP_URL"],
                        "-O", os.path.join(img_dir, "test_images.zip")], check=True)
        subprocess.run(["unzip", "-o", "-q", os.path.join(img_dir, "test_images.zip"),
                        "-d", img_dir], check=True)
        # flatten one nesting level if the zip unpacked into a subfolder
        inner = glob.glob(os.path.join(img_dir, "*", "*.*"))
        if inner and len(glob.glob(os.path.join(img_dir, "*.*"))) < n:
            for f in inner:
                os.rename(f, os.path.join(img_dir, os.path.basename(f)))


def load_test_image_paths(img_dir: str, n: int) -> list[str]:
    """The n=200 test images in the EXACT order of the EEG file.

    Order source, in priority:
      1. image_metadata.npy (official; maps EEG image index -> THINGS filename)
      2. sorted files in img_dir (FALLBACK — verify with --check!)
    """
    meta_path = CONFIG["IMAGE_METADATA_NPY"]
    if os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True)
        if isinstance(meta, np.ndarray) and meta.dtype == object and meta.shape == ():
            meta = meta.item()
        if hasattr(meta, "keys"):
            names = None
            for key in ("test_img_files", "test_image_ids", "test", "images"):
                if key in meta:
                    names = list(meta[key])[:n]
                    break
            if names is None:
                raise KeyError(f"image_metadata.npy keys: {list(meta.keys())}")
            # resolve by basename anywhere under img_dir (files may be nested
            # in concept subfolders, e.g. 00013_aircraft_carrier/*.jpg)
            all_files = glob.glob(os.path.join(img_dir, "**", "*.*"), recursive=True)
            by_base = {os.path.basename(f): f for f in all_files}
            out = []
            for x in names:
                base = os.path.basename(str(x).lstrip("./"))
                if base in by_base:
                    out.append(by_base[base])
                else:  # concept_name prefix match fallback
                    stem = os.path.splitext(base)[0]
                    cands = [f for b, f in by_base.items() if stem in b]
                    out.append(cands[0] if cands else os.path.join(img_dir, base))
            return out
    files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    if len(files) < n:
        raise FileNotFoundError(
            f"only {len(files)} images in {img_dir}; need {n}. "
            f"Set TEST_IMAGES_ZIP_URL or IMG_DIR.")
    return files[:n]

