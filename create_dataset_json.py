


base_path = 'data\\fairface-img-margin025-trainval\\val'

# create the following json file named val.json
# [{"image_file": "1.png"}] for each image in the val folder
# this is the format that the dataset reader expects
import os
import json
files = os.listdir(base_path)

data = []
for file in files:
    file = os.path.join(base_path, file)
    data.append(file)

with open(base_path.split(os.sep)[-1] + '.json', 'w') as f:
    json.dump(data, f)