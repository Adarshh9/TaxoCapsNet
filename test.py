# ------------------------------------------------------------
# TaxoSHAP API end-to-end test script (Enhanced with error handling)
# ------------------------------------------------------------
import requests, json, base64, os, io
from PIL import Image

NGROK_URL = "https://5015276b3329.ngrok-free.app"  # ◀️  update if tunnel restarts
HEADERS = {"Content-Type": "application/json"}

def pretty(js):
    print(json.dumps(js, indent=2)[:1200] + (" ..." if len(json.dumps(js))>1200 else ""))

def safe_request(method, url, **kwargs):
    """Make request with error handling"""
    try:
        if method.upper() == 'GET':
            r = requests.get(url, **kwargs)
        elif method.upper() == 'POST':
            r = requests.post(url, **kwargs)
        
        print(f"Status: {r.status_code}")
        if r.status_code != 200:
            print(f"Error response: {r.text}")
            return None
        return r.json()
    except Exception as e:
        print(f"Request failed: {e}")
        return None

# 1) HEALTH
print("\n🩺 /health")
health_data = safe_request('GET', f"{NGROK_URL}/health")
if health_data:
    pretty(health_data)

# 2) SAMPLES
print("\n📋 /samples")
samples_data = safe_request('GET', f"{NGROK_URL}/samples")
if samples_data and samples_data.get('status') == 'success':
    samples = samples_data.get('samples', {})
    pretty(samples)
    first_sample_id = next(iter(samples)) if samples else None
else:
    print("Failed to get samples")
    first_sample_id = None

if not first_sample_id:
    print("❌ No samples available, exiting")
    exit()

# 3) SAMPLE INFO
print(f"\n🔎 /samples/{first_sample_id}/info")
sample_info = safe_request('GET', f"{NGROK_URL}/samples/{first_sample_id}/info")
if sample_info:
    pretty(sample_info)

# 4) PHYLUMS
print("\n🧬 /phylums")
phylums_data = safe_request('GET', f"{NGROK_URL}/phylums")
if phylums_data:
    pretty(phylums_data)

# 5) MODEL INFO
print("\n🖥️ /model/info")
model_info = safe_request('GET', f"{NGROK_URL}/model/info")
if model_info:
    pretty(model_info)

# 6) ANALYZE
payload = {
    "sample_id": first_sample_id,
    "method": "tree",
    "max_display": 15,
    "n_background": 30
}
print(f"\n🚀 POST /analyze → {payload}")
analysis_data = safe_request('POST', f"{NGROK_URL}/analyze", headers=HEADERS, json=payload)

if analysis_data and analysis_data.get('status') == 'success':
    # Print key analysis info
    key_info = {}
    for key in ["status", "analysis_id", "sample_id", "real_test_index", "method"]:
        if key in analysis_data:
            key_info[key] = analysis_data[key]
    pretty(key_info)
    
    # Print prediction results
    if 'prediction_results' in analysis_data:
        print("\n🔮 Prediction Results:")
        pred_results = analysis_data['prediction_results']
        print(f"   Predicted Class: {pred_results.get('predicted_class', 'N/A')}")
        print(f"   Probability: {pred_results.get('prediction_probability', 'N/A'):.4f}")
        print(f"   True Label: {pred_results.get('true_label', 'N/A')}")
        print(f"   Correct: {pred_results.get('prediction_correct', 'N/A')}")
    
    # 7) SAVE VISUALIZATIONS
    viz_folder = "taxoshap_visuals"
    os.makedirs(viz_folder, exist_ok=True)
    
    for name, b64 in analysis_data.get("visualizations", {}).items():
        try:
            img_data = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_data))
            filename = os.path.join(viz_folder, f"{analysis_data['analysis_id']}_{name}.png")
            img.save(filename)
            print(f"📁 saved → {filename}")
        except Exception as e:
            print(f"⚠️ could not save {name}: {e}")
else:
    print("❌ Analysis failed or returned error")
    if analysis_data:
        print(f"Error: {analysis_data.get('message', 'Unknown error')}")

print("\n✅ API test finished.")


