import io
import json
from typing import Dict

import google.generativeai as genai
from PIL import Image

from config import GEMINI_API_KEY, GEMINI_MODEL_VISION, require_api_key


def extract_visual_elements(png_bytes: bytes) -> Dict[str, str]:
    require_api_key()
    genai.configure(api_key=GEMINI_API_KEY)

    image = Image.open(io.BytesIO(png_bytes))
    model = genai.GenerativeModel(GEMINI_MODEL_VISION)

    prompt = (
        "Extract equations, tables, and figure captions from this PDF page. "
        "Return JSON with keys: equations, tables, figures. Use concise markdown."
    )

    response = model.generate_content([prompt, image])
    text = response.text.strip() if response.text else ""

    try:
        data = json.loads(text)
        return {
            "equations": str(data.get("equations", "")),
            "tables": str(data.get("tables", "")),
            "figures": str(data.get("figures", "")),
        }
    except json.JSONDecodeError:
        return {"equations": "", "tables": "", "figures": text}
