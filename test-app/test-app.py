from flask import Flask, jsonify, Response, request
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('flask_requests_total', 'Total number of requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('flask_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'])

@app.before_request
def before_request():
    app.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - app.start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.endpoint or 'unknown',
        status=response.status_code
    ).inc()
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.endpoint or 'unknown'
    ).observe(duration)
    return response

@app.route('/')
def home():
    return jsonify({
        "code": "200",
        "status":"OK"
    })

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)