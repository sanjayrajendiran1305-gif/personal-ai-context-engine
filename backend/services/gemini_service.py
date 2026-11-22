import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)
_client = None


def get_client():
    global _client
    if _client is None:
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Gemini API key is missing. Add GEMINI_API_KEY to your .env file.")
        _client = genai.Client(api_key=key)
    return _client


def is_configured():
    return bool(os.getenv("GEMINI_API_KEY"))


def get_model():
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def generate_response(prompt, images=None):
    client = get_client()
    contents = [prompt]
    if images:
        from google.genai import types
        for image in images:
            contents.append(types.Part.from_bytes(data=image["data"], mime_type=image["mime_type"]))
    response = client.models.generate_content(model=get_model(), contents=contents)
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned no text response. Check that the selected model supports this input.")
    return text


def generate_image(prompt):
    model = os.getenv("GEMINI_IMAGE_MODEL", "").strip()
    if not model:
        raise RuntimeError("Image generation is not configured. Set GEMINI_IMAGE_MODEL to an image-generation model available to your API key.")
    client = get_client()
    from google.genai import types
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )
    for part in getattr(response, "parts", []) or []:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data, getattr(inline, "mime_type", "image/png")
    raise RuntimeError("The configured image model did not return an image. Check the model name and API access.")
