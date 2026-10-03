import os
import json
import hashlib
from google import genai
from google.genai import types

# Session cache: key = {hash}_{model}, value = structured result dict
_GATE_CACHE = {}

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gemini_config.json")

def _load_initial_config():
    cfg = {
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "model": "gemini-3.5-flash-lite"
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("api_key"):
                    cfg["api_key"] = data["api_key"]
                if data.get("model"):
                    cfg["model"] = data["model"]
        except Exception as e:
            print(f"Warning: could not read {CONFIG_FILE}: {e}")
    return cfg

# Current configuration
config = _load_initial_config()

def save_config(api_key: str = None, model: str = None):
    if api_key is not None:
        config["api_key"] = api_key
    if model is not None:
        config["model"] = model
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "api_key": config["api_key"],
                "model": config["model"]
            }, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save {CONFIG_FILE}: {e}")

def clear_config():
    config["api_key"] = ""
    try:
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
    except Exception as e:
        print(f"Warning: could not remove {CONFIG_FILE}: {e}")

def get_image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()

def test_connection(api_key: str, model: str) -> dict:
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents="Respond with OK if you are working."
        )
        if response.text:
            return {"status": "CONNECTED"}
        return {"status": "CONNECTION FAILED", "error": "Empty response"}
    except Exception as e:
        return {"status": "CONNECTION FAILED", "error": str(e)}

def analyze_image_validity(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    if not config["api_key"]:
        raise ValueError("Missing API key. Please configure the Gemini Vision Gate.")
        
    img_hash = get_image_hash(image_bytes)
    cache_key = f"{img_hash}_{config['model']}"
    
    if cache_key in _GATE_CACHE:
        return _GATE_CACHE[cache_key]

    client = genai.Client(api_key=config["api_key"])
    
    # Prompt instructing the model on its role
    prompt = """
    You are the GEMINI VISION GATE, an image-validity gate and visual assessment engine for a skin-lesion analysis application.
    Your job is to analyze the uploaded image, determine whether it is suitable for downstream lesion analysis, and perform a structured visual assessment of the visible lesion.
    Do NOT diagnose melanoma or calculate cancer probability.
    
    Analyze validity:
    - whether relevant human skin is visible
    - whether a lesion is visible and can reasonably be localized
    - whether the image is an actual photographic input and not an artificial drawing
    - whether the image is irrelevant (e.g. cat, dog, object, landscape)
    - whether the lesion is obscured or too blurry
    
    Your decision MUST be one of: REJECT, RETAKE, CONTINUE, REVIEW.
    
    Visual Assessment (Only if decision is CONTINUE or REVIEW):
    Analyze observable visual characteristics such as shape, symmetry, border regularity, edge smoothness, roundness, color variation, texture variation, and visual complexity.
    Use a 0-100 scale:
    - shape_score (higher = more regular)
    - symmetry_score (higher = more symmetric)
    - border_regularity_score (higher = smoother/more regular border)
    - edge_smoothness_score (higher = smoother edge)
    - roundness_score (higher = more circular/rounded)
    - color_variation_score (higher = greater visible color variation)
    - texture_variation_score (higher = greater visible texture variation)
    - visual_complexity_score (higher = greater structural complexity)
    - overall_morphology_score (0 = very simple/regular morphology, 100 = highly irregular/complex morphology). This is NOT a cancer probability.
    
    Provide concise, human-readable observations describing only what is visually observable. Do NOT mention melanoma, cancer, or malignancy.
    """
    
    try:
        response = client.models.generate_content(
            model=config['model'],
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "decision": types.Schema(
                            type=types.Type.STRING,
                            enum=["REJECT", "RETAKE", "CONTINUE", "REVIEW"],
                        ),
                        "image_valid": types.Schema(type=types.Type.BOOLEAN),
                        "scene": types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "contains_skin": types.Schema(type=types.Type.BOOLEAN),
                                "contains_person": types.Schema(type=types.Type.BOOLEAN),
                                "contains_face": types.Schema(type=types.Type.BOOLEAN),
                                "contains_drawing": types.Schema(type=types.Type.BOOLEAN),
                                "contains_tattoo_or_artwork": types.Schema(type=types.Type.BOOLEAN),
                                "is_photographic": types.Schema(type=types.Type.BOOLEAN),
                            }
                        ),
                        "lesion": types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "visible": types.Schema(type=types.Type.BOOLEAN),
                                "clearly_localized": types.Schema(type=types.Type.BOOLEAN),
                                "appears_artificial": types.Schema(type=types.Type.BOOLEAN),
                                "appearance": types.Schema(
                                    type=types.Type.STRING,
                                    enum=["skin_lesion", "normal_skin", "artificial", "unknown"],
                                ),
                            }
                        ),
                        "quality": types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "blur": types.Schema(
                                    type=types.Type.STRING,
                                    enum=["low", "medium", "high"],
                                ),
                                "lighting": types.Schema(
                                    type=types.Type.STRING,
                                    enum=["good", "acceptable", "poor"],
                                ),
                                "occlusion": types.Schema(
                                    type=types.Type.STRING,
                                    enum=["none", "mild", "severe"],
                                ),
                                "resolution": types.Schema(
                                    type=types.Type.STRING,
                                    enum=["adequate", "inadequate"],
                                ),
                                "lesion_visibility": types.Schema(
                                    type=types.Type.INTEGER, 
                                    description="Scale from 0 to 100",
                                ),
                            }
                        ),
                        "suspicion": types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "suspicious_input": types.Schema(type=types.Type.BOOLEAN),
                                "reason": types.Schema(type=types.Type.STRING),
                            }
                        ),
                        "visual_assessment": types.Schema(
                            type=types.Type.OBJECT,
                            nullable=True,
                            properties={
                                "shape_score": types.Schema(type=types.Type.INTEGER),
                                "symmetry_score": types.Schema(type=types.Type.INTEGER),
                                "border_regularity_score": types.Schema(type=types.Type.INTEGER),
                                "edge_smoothness_score": types.Schema(type=types.Type.INTEGER),
                                "roundness_score": types.Schema(type=types.Type.INTEGER),
                                "color_variation_score": types.Schema(type=types.Type.INTEGER),
                                "texture_variation_score": types.Schema(type=types.Type.INTEGER),
                                "visual_complexity_score": types.Schema(type=types.Type.INTEGER),
                                "overall_morphology_score": types.Schema(type=types.Type.INTEGER),
                                "confidence": types.Schema(type=types.Type.INTEGER),
                                "observations": types.Schema(
                                    type=types.Type.ARRAY,
                                    items=types.Schema(type=types.Type.STRING)
                                )
                            }
                        ),
                        "visual_thresholds": types.Schema(
                            type=types.Type.OBJECT,
                            nullable=True,
                            properties={
                                "morphology": types.Schema(type=types.Type.STRING, enum=["low", "moderate", "high"]),
                                "border": types.Schema(type=types.Type.STRING, enum=["regular", "mildly_irregular", "irregular"]),
                                "symmetry": types.Schema(type=types.Type.STRING, enum=["symmetric", "mildly_asymmetric", "asymmetric"]),
                                "visual_concern": types.Schema(type=types.Type.STRING, enum=["low", "moderate", "elevated"])
                            }
                        ),
                        "reason": types.Schema(type=types.Type.STRING),
                        "lesion_bbox": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.INTEGER),
                            description="Optional [ymin, xmin, ymax, xmax] coordinates from 0 to 1000",
                            nullable=True
                        )
                    }
                ),
                temperature=0.1
            )
        )
        
        result_json = response.text
        result_dict = json.loads(result_json)
        
        # Ensure decision is one of the valid options (defensive)
        if result_dict.get("decision") not in ["REJECT", "RETAKE", "CONTINUE", "REVIEW"]:
            result_dict["decision"] = "REVIEW"
            
        _GATE_CACHE[cache_key] = result_dict
        return result_dict
        
    except Exception as e:
        raise Exception(f"AI_GATE_UNAVAILABLE: {str(e)}")
