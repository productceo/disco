from urllib.request import urlopen
import numpy as np
import requests
import json
import os
import re

# Imageset Diversity Score
# Documentation: https://1drv.ms/w/s!AkQpwGOlM3EpgSGwPW24y9DUHKJ7

class Disco:

    # Init Function
    def __init__(self):
        self.facenet_server = ""
        self.image_download_path = ""
        self.input_file = ""
        self.output_file = ""
        self.avatar_file = ""
        self.avatar_queries = set()
        self.filename_length = 8

    # Microsoft Internal Use: Converting Avatar output file to Disco input file format
    def convert_avatar_file(self):
        avatar_data = list()
        with open(self.avatar_file, encoding="ISO-8859-1") as avatar_file:
            next(avatar_file)
            for line in avatar_file:
                avatar_datum = re.split("\t", line)

                search_engine = ''.join(avatar_datum[1].split(" "))
                search_query = avatar_datum[3]
                imageset = "{}-{}".format(search_engine, search_query)

                if search_query not in self.avatar_queries:
                    continue

                avatar_data.append({
                    "image_uri": avatar_datum[7],
                    "imageset": imageset
                })

        with open(self.input_file, "w+") as disco_file:
            for avatar_datum in avatar_data:
                disco_datum = {
                    "image_uri": avatar_datum["image_uri"],
                    "imageset": avatar_datum["imageset"]
                }
                disco_file.write(json.dumps(disco_datum) + "\n")

    # Downloading images
    def format_filename(self, image_count):
        filename = str(image_count)
        while len(filename) < self.filename_length:
            filename = "0{}".format(filename)
        return filename

    def download_image(self, image_data, image_count_per_imageset):
        if image_data["imageset"] not in image_count_per_imageset.keys():
            image_count_per_imageset[image_data["imageset"]] = 0
        image_count = image_count_per_imageset[image_data["imageset"]]

        directory = self.image_download_path
        subsdirectory = image_data["imageset"]
        filename = self.format_filename(image_count)

        imageset_directory = "{}/{}".format(directory, subsdirectory)
        image_file_path = "{}/{}.jpg".format(imageset_directory, filename)

        if (not os.path.exists(imageset_directory)):
            os.system("mkdir {}".format(imageset_directory))

        try:
            response = urlopen(image_data["image_uri"])
            responseHtml = response.read()
            image_file = open(image_file_path, "wb")
            image_file.write(responseHtml)
            image_file.close()
            image_count_per_imageset[image_data["imageset"]] += 1
        except:
            pass

        return image_count_per_imageset

    def download_images(self):
        image_count_per_imageset = dict()

        if (not os.path.exists(self.image_download_path)):
            os.system("mkdir {}".format(self.image_download_path))

        with open(self.input_file, "r") as disco_file:
            for line in disco_file:
                image_data = json.loads(line)
                image_count_per_imageset = self.download_image(
                    image_data, image_count_per_imageset)

    # Generating face embedding vectors
    def facenet(self, image):
        files = {'image': ('image.jpg', open(image, 'rb'), 'images/jpeg')}
        r = requests.post(self.facenet_server, files=files).json()
        return r

    def generate_face_vectors(self, imageset):
        face_vectors = list()
        images = [f for f in os.listdir(
            imageset) if os.path.isfile(os.path.join(imageset, f))]
        for image in images:
            face_vector = self.facenet("{}/{}".format(imageset, image))
            if "predictions" in face_vector.keys():
                for prediction in face_vector["predictions"]:
                    face_vectors.append(prediction["embedding"])
        return face_vectors

    # Calculating imageset diversity score
    def generate_permutation_pairs(self, elements):
        permutations = list()
        length = len(elements)
        for i in range(length):
            for j in range(length):
                if (i != j):
                    permutations.append([elements[i], elements[j]])
        return permutations

    def calculate_vector_distance(self, vector1, vector2):
        return np.linalg.norm(np.asarray(vector1) - np.asarray(vector2))

    def calculate_average_pairwise_distance(self, vector_pairs):
        sum_pairwise_distance = 0
        for vector_pair in vector_pairs:
            sum_pairwise_distance += self.calculate_vector_distance(
                vector_pair[0], vector_pair[1])
        average_pairwise_distance = sum_pairwise_distance / len(vector_pairs)
        return average_pairwise_distance

    def calculate_imageset_diversity_score(self):
        imagesets = [
            os.path.join(self.image_download_path, o)
            for o in os.listdir(self.image_download_path)
            if os.path.isdir(os.path.join(self.image_download_path, o))
        ]

        with open(self.output_file, "w+") as output_file:
            for imageset in imagesets:
                face_vectors = self.generate_face_vectors(imageset)
                face_vector_pairs = self.generate_permutation_pairs(
                    face_vectors)
                diversity_score = self.calculate_average_pairwise_distance(
                    face_vector_pairs)

                diversity_score_data = {
                    "imageset": imageset.rsplit("/", 1)[-1],
                    "diversity_score": diversity_score
                }

                output_file.write(json.dumps(diversity_score_data) + "\n")

    def print_imageset_diversity_score(self):
        with open(self.output_file, "r+") as output_file:
            for line in output_file:
                data = json.loads(line)
                print("Imageset: {} | Diversity Score: {}".format(
                    data["imageset"],
                    data["diversity_score"]
                ))
