
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from brain.learner import AutonomousAI
from crawler.web_crawler import WebCrawler
import threading
import time

app = Flask(__name__)
CORS(app)

# Initialize AI
print("🚀 Starting AI...")
ai = AutonomousAI()
crawler = WebCrawler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    # AI thinks and responds
    response = ai.generate_response(user_message)
    
    # Background learning
    ai.learn_from_text(user_message, source="user")
    
    return jsonify({
        'response': response,
        'stats': ai.get_stats()
    })

@app.route('/api/learn', methods=['POST'])
def learn():
    """Teach AI directly"""
    data = request.json
    text = data.get('text', '')
    topic = data.get('topic', 'general')
    
    ai.memory.store_knowledge(topic, text, source="manual", importance=1.0)
    ai.learn_from_text(text)
    
    return jsonify({'status': 'learned', 'topic': topic})

@app.route('/api/browse', methods=['POST'])
def browse():
    """Make AI browse internet"""
    data = request.json
    query = data.get('query', '')
    
    def crawl_task():
        results = crawler.search_and_learn(query, ai.memory, max_pages=3)
        for result in results:
            ai.learn_from_text(result['content'], source=result['source'])
    
    # Run in background
    thread = threading.Thread(target=crawl_task)
    thread.start()
    
    return jsonify({
        'status': 'browsing_started',
        'query': query,
        'message': f'🔍 Searching for "{query}" and learning...'
    })

@app.route('/api/stats')
def stats():
    return jsonify(ai.get_stats())

@app.route('/api/memories')
def memories():
    """Show what AI remembers"""
    topic = request.args.get('topic')
    mems = ai.memory.recall(topic=topic, limit=20)
    return jsonify({
        'memories': [
            {'topic': m[0], 'content': m[1][:200], 'source': m[2], 'importance': m[3]}
            for m in mems
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
