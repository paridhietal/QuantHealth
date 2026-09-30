"""
extract_api.py — pull clinical values out of an uploaded report
=====================================================================

Matches the field keys in lib/patientFields.ts as delivered by Person A:
- heart_disease: RAW UCI codes (cp 1-4, slope 1-3, thal 3/6/7)
- breast_cancer: all 30 features (10 mean + 10 se + 10 worst)
- parkinsons: all 22 voice measures
- diabetes: not exported yet — this endpoint returns 501 for it

Only continuous, unambiguous fields are auto-extracted. Categorical fields
with non-obvious raw codes (cp, fbs, restecg, exang, slope, ca, thal) are
SKIPPED on purpose — a report almost never prints "cp: 3" as a labeled
line, and guessing wrong on a coded categorical is worse than leaving it
blank for the person to fill in. 'sex' is handled specially (male/female
-> 1/0) since that one's safe and common.

Extraction never auto-submits a prediction — see extracted/not_found in
the response and let the frontend prefill + let the person review.

DEPENDENCIES:
    pip install pdfplumber pytesseract pillow
    # pytesseract also needs the Tesseract binary on the OS:
    # https://github.com/tesseract-ocr/tesseract#installing-tesseract
"""

import io
import re
from typing import Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()

VALID = ["breast_cancer", "diabetes", "heart_disease", "parkinsons"]

# Only fields listed here are attempted. Missing key = not auto-extracted
# (categorical/coded fields the person fills in manually).
# Value = list of phrases to search for; auto-generated ones are added too.
MANUAL_SYNONYMS: Dict[str, Dict[str, List[str]]] = {
    "heart_disease": {
        "age": ["age"],
        "trestbps": ["resting blood pressure", "resting bp", "blood pressure", "bp"],
        "chol": ["serum cholesterol", "total cholesterol", "cholesterol"],
        "thalach": ["max heart rate", "maximum heart rate", "peak heart rate"],
        "oldpeak": ["st depression"],
        # sex handled separately via find_sex_value
        # cp, fbs, restecg, exang, slope, ca, thal: intentionally skipped (coded categoricals)
    },
    "parkinsons": {
        "mdvp_jitter_pct": ["jitter(%)", "jitter %", "mdvp:jitter(%)"],
        "mdvp_jitter_abs": ["jitter(abs)", "jitter absolute"],
        "mdvp_shimmer_db": ["shimmer(db)", "shimmer db"],
    },
    "breast_cancer": {},  # fully auto-generated from key names, see below
    "diabetes": {},  # not exported yet
}

AUTO_EXTRACT_KEYS: Dict[str, List[str]] = {
    "heart_disease": ["age", "trestbps", "chol", "thalach", "oldpeak"],  # + sex handled separately
    "parkinsons": [
        "mdvp_fo", "mdvp_fhi", "mdvp_flo", "mdvp_jitter_pct", "mdvp_jitter_abs",
        "mdvp_rap", "mdvp_ppq", "jitter_ddp", "mdvp_shimmer", "mdvp_shimmer_db",
        "shimmer_apq3", "shimmer_apq5", "mdvp_apq", "shimmer_dda",
        "nhr", "hnr", "rpde", "dfa", "spread1", "spread2", "d2", "ppe",
    ],
    "breast_cancer": [
        f"{base}_{suffix}"
        for suffix in ("mean", "se", "worst")
        for base in (
            "radius", "texture", "perimeter", "area", "smoothness",
            "compactness", "concavity", "concave_points", "symmetry",
            "fractal_dimension",
        )
    ],
    "diabetes": [],
}

SEX_MALE_WORDS = {"male", "m"}
SEX_FEMALE_WORDS = {"female", "f"}


def phrases_for_key(dataset: str, key: str) -> List[str]:
    manual = MANUAL_SYNONYMS.get(dataset, {}).get(key, [])
    auto = key.replace("_", " ")
    return list(dict.fromkeys([*manual, auto]))  # dedupe, keep order


def extract_text_from_pdf(data: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise HTTPException(status_code=501, detail="PDF support needs 'pdfplumber' (pip install pdfplumber).")
    text_parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    text = "\n".join(text_parts)
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text in this PDF — it may be a scanned image. Try uploading it as a JPG/PNG instead.",
        )
    return text


def extract_text_from_image(data: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise HTTPException(status_code=501, detail="Image OCR needs 'pytesseract' + 'pillow', plus the Tesseract binary on the OS.")
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read this file as an image.")
    text = pytesseract.image_to_string(img)
    if not text.strip():
        raise HTTPException(status_code=422, detail="OCR found no readable text in this image.")
    return text


def find_numeric_value(text: str, phrases: List[str]) -> Optional[float]:
    for phrase in phrases:
        # allow flexible punctuation/spacing between words (e.g. "MDVP:Fo" for "mdvp fo")
        loose = r"[\s:]*".join(re.escape(w) for w in phrase.split())
        pattern = rf"{loose}\s*[:\-]?\s*([\d]+\.?[\d]*)"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def find_sex_value(text: str) -> Optional[str]:
    m = re.search(r"(?:sex|gender)\s*[:\-]?\s*(\w+)", text, re.IGNORECASE)
    if not m:
        return None
    word = m.group(1).strip().lower()
    if word in SEX_MALE_WORDS:
        return "1"
    if word in SEX_FEMALE_WORDS:
        return "0"
    return None


@router.post("/datasets/{dataset}/extract")
async def extract(dataset: str, file: UploadFile = File(...)):
    if dataset not in VALID:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset}'.")
    if dataset == "diabetes":
        raise HTTPException(status_code=501, detail="Diabetes hasn't been exported yet — no field mapping available.")

    data = await file.read()
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()

    if "pdf" in content_type or filename.endswith(".pdf"):
        text = extract_text_from_pdf(data)
    elif content_type.startswith("image/") or filename.endswith((".jpg", ".jpeg", ".png", ".webp")):
        text = extract_text_from_image(data)
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload a PDF or an image (JPG/PNG).")

    keys = AUTO_EXTRACT_KEYS.get(dataset, [])
    extracted: Dict[str, str] = {}
    not_found: List[str] = []

    if dataset == "heart_disease":
        sex_val = find_sex_value(text)
        if sex_val is not None:
            extracted["sex"] = sex_val
        else:
            not_found.append("sex")

    for key in keys:
        val = find_numeric_value(text, phrases_for_key(dataset, key))
        if val is not None:
            extracted[key] = str(val)
        else:
            not_found.append(key)

    return {
        "extracted": extracted,
        "not_found": not_found,
        "text_preview": text[:500],
    }