from flask import Flask, request, jsonify, render_template
from supervised import SmartAssistant  # Import the SmartAssistant class

# Initialize Flask app
app = Flask(__name__)

# Initialize the SmartAssistant instance
assistant = SmartAssistant(data_dir="data")

@app.route('/')
def index():
    """Render the main chatbot interface."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handle user questions and return chatbot responses."""
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'answer': 'Please ask a question.'}), 400
    
    # Use the SmartAssistant to generate an answer
    answer = assistant.answer(question)
    return jsonify({'answer': answer})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)