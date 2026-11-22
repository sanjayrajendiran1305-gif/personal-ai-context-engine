import base64
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)
_client = None


def get_client():
    global _client
    if _client is None:
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Groq API key is missing. Add GROQ_API_KEY to your .env file.")
        _client = Groq(api_key=key)
    return _client


def is_configured():
    return bool(os.getenv("GROQ_API_KEY"))


def get_model():
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def generate_response(prompt, images=None):
    client = get_client()
    model = get_model()
    content = prompt
    if images:
        vision_model = os.getenv("GROQ_VISION_MODEL", "").strip()
        if not vision_model:
            raise RuntimeError("The configured Groq model is text-only. Set GROQ_VISION_MODEL to a vision-capable model to analyze images with Groq.")
        model = vision_model
        content = [{"type": "text", "text": prompt}]
        for image in images:
            encoded = base64.b64encode(image["data"]).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image['mime_type']};base64,{encoded}"},
            })
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.4,
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("Groq returned an empty response.")
    return answer
