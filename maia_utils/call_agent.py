import openai
import anthropic
from openai.error import RateLimitError, ServiceUnavailableError, APIError, InvalidRequestError

import time
from random import random
import warnings
import traceback

# new
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import base64

warnings.filterwarnings("ignore")

# User inputs:
# Load your API key from an environment variable or secret management service
# openai.api_key = os.getenv("OPENAI_API_KEY")
# OR 
# Load your API key manually:
# openai.api_key = API_KEY

# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.organization = os.getenv("OPENAI_ORGANIZATION")
# anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_content_from_message(message):
    """
    Extract content from a message, handling both simple string content
    and structured content with text/image_url types, with token optimization.
    """
    # Handle array of content parts
    parts = []
    if isinstance(message["content"], str):
        parts.append({
            "type": "text",
            "text": message["content"]
        })
        return parts
        
    for content in message["content"]:
        if content["type"] == "text":
            parts.append(content)
        elif content["type"] == "image_url":
            image_url = content["image_url"]["url"]
            if image_url.startswith("data:image"):
                header, base64_data = image_url.split(',', 1)
                if base64_data.startswith('iVBOR'):
                    media_type = 'image/png'
                else:
                    media_type = header.split(':')[1].split(';')[0]
                
                image_content = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    }
                }
                parts.append(image_content)
    return parts

def system_blocks_to_str(blocks):
    # blocks: str OR [{"type":"text","text":...}, ...]
    if isinstance(blocks, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) and b.get("type") in ("text", "input_text")
            else str(b)
            for b in blocks
        )
    return str(blocks)

def blocks_to_parts_gemini(blocks):
    # blocks: str OR list of {"type": "...", ...} as produced by get_content_from_message
    if isinstance(blocks, str):
        return [types.Part.from_text(text=blocks)]

    parts = []
    for b in blocks:
        if isinstance(b, dict):
            t = b.get("type")
            if t in ("text", "input_text"):
                parts.append(types.Part.from_text(text=b.get("text", "")))
            elif t == "image":
                # your function builds: {"type":"image","source":{"type":"base64","media_type":..., "data":...}}
                src = b.get("source", {})
                if src.get("type") == "base64":
                    data_b64 = src.get("data", "")
                    if data_b64:
                        parts.append(
                            types.Part.from_bytes(
                                data=base64.b64decode(data_b64),
                                mime_type=src.get("media_type", "image/jpeg"),
                            )
                        )
            elif t == "image_url":
                # If you ever leave these as URLs instead of converting to base64
                url = (b.get("image_url") or {}).get("url") or b.get("url")
                if url:
                    parts.append(types.Part.from_uri(file_uri=url, mime_type=b.get("mime_type")))
        elif isinstance(b, str):
            parts.append(types.Part.from_text(text=b))
        else:
            parts.append(types.Part.from_text(text=str(b)))
    return parts

def ask_agent(model, history):
    max_retries = 5
    count = 0
    while count < max_retries:
        try:
            if model.startswith('gemini'):
                # Example: 'gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-pro', etc.
                client = genai.Client()  # reads GEMINI_API_KEY from env

                system_instruction = None
                contents = []
                for msg in history:
                    raw = get_content_from_message(msg)
                    if msg["role"] == "system":
                        system_instruction = system_blocks_to_str(raw)
                        continue
                    # Gemini roles are "user" and "model"
                    g_role = "user" if msg["role"] == "user" else "model"
                    contents.append(types.Content(role=g_role, parts=blocks_to_parts_gemini(raw)))

                cfg = types.GenerateContentConfig(
                    max_output_tokens=4096,
                    system_instruction=system_instruction if system_instruction else None,
                )

                resp = client.models.generate_content(
                    model=model,          # pass through the requested Gemini model
                    contents=contents,    # built from history
                    config=cfg,
                )
                return resp.text

            else:
                print(f"Unrecognized model name: {model}")
                return None

        except (openai.error.RateLimitError, 
                openai.error.ServiceUnavailableError, 
                openai.error.APIError,
                anthropic.RateLimitError,
                genai_errors.APIError) as e:
            count += 1
            print(f'API error: {str(e)}')
            # Optional: only retry for transient Gemini errors
            # if isinstance(e, genai_errors.APIError) and e.code not in (429, 500, 502, 503, 504):
            #     raise
            wait_time = 60 + 10*random()
            print(f"Attempt {count}/{max_retries}. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            traceback.print_exc()
            return None

    return None