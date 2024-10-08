import requests
import json

url = "http://localhost:4000/entity"
data = {
    "sentence": "Microsoft is based in the United States of America.",
}
response = requests.post(url, json=data)

json_response = response.json()
# print(json.dumps(json_response, indent=4))

detected_entities = json_response["merged_entities"]
for entity in detected_entities:
    entity_name, entity_type = entity
    print(f"{entity_name} = {entity_type}")
