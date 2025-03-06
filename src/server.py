from flask import Flask, request, jsonify
from agents.coordinator import TaskCoordinator

app = Flask(__name__)
coordinator = TaskCoordinator()

@app.route('/task', methods=['POST'])
def submit_task():
    data = request.get_json()
    task = data.get('task')
    if not task:
        return jsonify({'error': 'No task provided'}), 400
    
    result = coordinator.execute_task(task)
    return jsonify({'status': 'success', 'result': result})

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 