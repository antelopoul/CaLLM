import json

# Function to transform the JSON data
def transform_json(input_data):
    data = json.loads(input_data)
    for annotation in data['annotations']:
        if isinstance(annotation['caption'], list):
            annotation['caption'] = ' '.join(annotation['caption'])
    return json.dumps(data, indent=4)

# Load JSON data from a file
with open("../avenue02/02_new.json", 'r') as file:
    input_data = file.read()

# Transform the JSON data
output_data = transform_json(input_data)

# Write the transformed JSON data to a file
with open('../avenue02/align02.json', 'w') as file:
    file.write(output_data)
