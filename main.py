import requests
from config import APIKEY

API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
HEADERS = {"Authorization": f"Bearer {APIKEY}"}
TOPICS = ["Sports", "Technology", "Business", "Politics", "Health"]

def classify(text):
    payload = {
        "inputs": {"text": text},
        "parameters": {"candidate_labels": TOPICS}
    }
    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
    if not r.ok:
        raise Exception(f"{r.status_code} - {r.text}")
    return r.json()

def bar(score):
    blocks = int(score * 10)
    return "█" * blocks + "░" * (10 - blocks)

def get_top(result):
    # Case 1: {"labels":[...], "scores":[...]}
    if isinstance(result, dict) and "labels" in result and "scores" in result:
        return result["labels"][0], float(result["scores"][0])

    # Case 2: [ {"label":..., "score":...}, ... ]
    if isinstance(result, list) and result and isinstance(result[0], dict) and "label" in result[0] and "score" in result[0]:
        best = max(result, key=lambda x: x["score"])
        return best["label"], float(best["score"])

    # If neither format matches, show what came back
    raise Exception(f"Unexpected response format: {result}")

while True:
    text = input("Enter a text to classify (exit to quit): ").strip()
    if text.lower() == "exit":
        break
    if not text:
        continue

    try:
        result = classify(text)
        label, score = get_top(result)
        print("\nTopic:", label)
        print("Confidence:", round(score * 100, 1), "%", bar(score), "\n")
    except Exception as e:
        print("Error:", e)
        print("Could not classify the text.\n")
