import json

# Example input JSON
input_json = "../avenue01/01_new.json"


# Function to transform the JSON data
def transform_json(input_data):
    data = json.loads(input_data)
    for annotation in data['annotations']:
        if isinstance(annotation['caption'], list):
            annotation['caption'] = annotation['caption'][0]
    return json.dumps(data, indent=4)

# Load JSON data from a file
with open(input_json, 'r') as file:
    input_data = file.read()

# Transform the JSON data
output_data = transform_json(input_data)

# Write the transformed JSON data to a file
with open('../avenue01/align01.json', 'w') as file:
    file.write(output_data)



