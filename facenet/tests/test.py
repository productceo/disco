#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import requests
import pytest


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'MAX Facial Recognizer'


def test_metadata():

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'max-facial-recognizer'
    assert metadata['name'] == 'MAX Facial Recognizer'
    assert metadata['description'] == 'Facial recognition model ' \
                                      'trained on LFW data to detect faces and ' \
                                      'generate embeddings'
    assert metadata['license'] == 'MIT'
    assert metadata['type'] == 'Facial Recognition'
    assert 'max-facial-recognizer' in metadata['source']


def test_predict():

    model_endpoint = 'http://localhost:5000/model/predict'

    # Test by the image with multiple faces
    img1_path = 'samples/Bryan.png'

    with open(img1_path, 'rb') as file:
        file_form = {'image': (img1_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200
    response = r.json()

    assert response['status'] == 'ok'
    assert len(response['predictions']) > 0
    assert len(response['predictions'][0]['detection_box']) == 4
    assert len(response['predictions'][0]['embedding']) == 512

    # Test by the image without faces
    img2_path = 'samples/IBM.jpeg'

    with open(img2_path, 'rb') as file:
        file_form = {'image': (img2_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200
    response = r.json()

    assert response['status'] == 'ok'
    assert len(response['predictions']) == 0

    # Test by the text data
    img3_path = 'samples/README.md'

    with open(img3_path, 'rb') as file:
        file_form = {'image': (img3_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__])
