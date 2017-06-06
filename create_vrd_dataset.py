import os
import urllib
import zipfile
import json
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt

def get_zip(url, zip_path, unzip_path):
    if not os.path.isfile(zip_path):
        urllib.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(unzip_path)

def get_dataset(data_dir):
    dataset_zip_path = os.path.join(data_dir, "json_dataset.zip")
    dataset_url = "http://cs.stanford.edu/people/ranjaykrishna/vrd/json_dataset.zip"
    get_zip(dataset_url, dataset_zip_path, data_dir)

    images_zip_path = os.path.join(data_dir, "sg_dataset.zip")
    images_url = "http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip"
    get_zip(images_url, images_zip_path, data_dir)

def convert_to_darkflow(data_dir, split):
    anns_dir = os.path.join(data_dir, "Annotations_%s" % split)

    with open(os.path.join(data_dir, "objects.json")) as f:
        objs_list = json.load(f)
    idx_to_name = {idx: name for idx, name in enumerate(objs_list)}

    if not os.path.isdir(anns_dir):
        os.mkdir(anns_dir)

    with open(os.path.join(data_dir, "annotations_%s.json" % split)) as f:
        data = json.load(f)

    cur_image_idx = 1
    for image_filename, image_data in data.items():
        image_filename = os.path.join("sg_%s_images" % split, image_filename)
        full_image_filename = os.path.join(data_dir, os.path.join("sg_dataset", image_filename))
        try:
            im = plt.imread(full_image_filename)
        except:
            continue

        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = image_filename
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "height").text = str(im.shape[0])
        ET.SubElement(size, "width").text = str(im.shape[1])
        ET.SubElement(size, "depth").text = str(im.shape[2])

        subject_data = [pair["subject"] for pair in image_data]
        object_data = [pair["object"] for pair in image_data]
        for info in subject_data + object_data:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = idx_to_name[info["category"]]
            bbox = ET.SubElement(obj, "bndbox")
            for field, val in zip(["ymin", "ymax", "xmin", "xmax"], info["bbox"]):
                ET.SubElement(bbox, field).text = str(val)

        tree = ET.ElementTree(root)
        tree.write(os.path.join(anns_dir, "%s.xml" % cur_image_idx))

        cur_image_idx += 1

if __name__ == "__main__":
    get_dataset("data")
    for split in ["train", "test"]:
        convert_to_darkflow("data", split)
