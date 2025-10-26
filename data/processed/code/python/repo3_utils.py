import requests

def make_api_call():
    headers = {"Authorization": "Bearer ghp_abcdefghijklmnopqrstuvwxyz123456"}
    response = requests.get("https://example.com/path")
    return response.json()
