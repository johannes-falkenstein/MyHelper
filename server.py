from flask import Flask, render_template, request, jsonify
import requests
from image_gen import generate_image

app = Flask(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image')
def image_page():
    return render_template('image.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_prompt = data.get("prompt", "")
    selected_model = data.get("model", "deepseek-r1:7b")  # Default model if not provided

    payload = {
        "model": selected_model,
        "prompt": user_prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response_data = response.json()

        print(f"Ollama API Response ({selected_model}):", response_data)

        if 'response' in response_data:
            return jsonify(response_data)
        else:
            return jsonify({"error": "Unexpected response format from Ollama", "details": response_data}), 500

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/generate_image', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    image_url = generate_image(prompt)
    return jsonify({"image_url": image_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)