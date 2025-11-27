import requests

# Generate code
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "def fibonacci(n):",
        "max_length": 150,
        "temperature": 0.7
    }
)

result = response.json()
print(f"Generated: {result['generated_text']}")
print(f"Time: {result['inference_time']}s")