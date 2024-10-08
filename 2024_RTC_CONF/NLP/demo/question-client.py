import requests
import json

url = "http://localhost:3000/question"
data = {
    "sentence": "The quick brown fox jumps over the lazy dog.",
}
response = requests.post(url, json=data)

json_response = response.json()
# print(json.dumps(json_response, indent=4))

print(f"statement: {json_response['question']}")
print(f"is_question: {json_response['is_question']}")
