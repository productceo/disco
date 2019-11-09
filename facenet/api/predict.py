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

from core.model import ModelWrapper
from flask_restplus import fields
from werkzeug.datastructures import FileStorage
from maxfw.core import MAX_API, PredictAPI

input_parser = MAX_API.parser()
# Example parser for file input
input_parser.add_argument('image', type=FileStorage, location='files',
                          required=True, help='An image file encoded as PNG, '
                                              'Tiff, or JPEG with an arbitrary '
                                              'size')

label_prediction = MAX_API.model('LabelPrediction', {
    'detection_box': fields.List(fields.Float(
        required=True, description='Bounding box for the detected face')),
    'probability': fields.Float(
        required=True, description='Probability of the detected face'),
    'embedding': fields.List(fields.Float(
        required=True, description='Embedding for the detected face'))
})

predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(
        required=True, description='Response status message'),
    'predictions': fields.List(fields.Nested(label_prediction),
                               description='Bounding boxes, probabilities, and '
                                           'embeddings for the detected faces')})


class ModelPredictAPI(PredictAPI):

    model_wrapper = ModelWrapper()

    @MAX_API.doc('predict')
    @MAX_API.expect(input_parser)
    @MAX_API.marshal_with(predict_response)
    def post(self):
        """Make a prediction given input data"""
        result = {'status': 'error'}

        args = input_parser.parse_args()
        input_data = args['image'].read()
        preds = self.model_wrapper.predict(input_data)
        result['predictions'] = preds
        result['status'] = 'ok'

        return result
