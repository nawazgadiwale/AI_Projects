from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load a free open-source chat model (this may take time on first run)
chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct")  # or zephyr-7b


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chatbot_route():
    data = request.get_json()
    user_message = data.get("message", "")

    try:
        prompt = f"User: {user_message}\nAssistant:"
        response = chatbot(prompt, max_new_tokens=150)
        reply = response[0]['generated_text'].split("Assistant:")[-1].strip()

        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
