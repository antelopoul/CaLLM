import json

# Load the data from the JSON file
with open('./avenue/07.json', 'r') as f:
    data = json.load(f)

# Add the new caption to the relevant objects
for obj in data:
    if 741 <= int(obj['image_id']) <= 900:
        obj['caption'].append('This is abnormal! Someone is moving strange to a wrong direction!')

# Write the updated data to a new JSON file
with open('./avenuex/07_new.json', 'w') as f:
    json.dump(data, f, indent=4)