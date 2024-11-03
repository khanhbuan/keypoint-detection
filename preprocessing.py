import os
import json

def write_info(data):
    data["licenses"] = [{'name': '', 'id': 0, 'url': ''}]
    data["info"] = {'contributor': '', 'date_created': '', 'description': '', 'url': '', 'version': '', 'year': ''}

img_list = []

path = "./annotations/person_keypoints_default.json"
with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)
    print(data["annotations"])
    for image in data["images"]:
        img_list.append(image["file_name"])

out = {}
out["licenses"] = [{'name': '', 'id': 0, 'url': ''}]
out["info"] = {'contributor': '', 'date_created': '', 'description': '', 'url': '', 'version': '', 'year': ''}
out["images"] = data["images"]