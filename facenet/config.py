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

# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False

# Application settings

# API metadata
API_TITLE = 'MAX Facial Recognizer'
API_DESC = 'Recognize faces in an image and extract embedding vectors for each face'
API_VERSION = '1.1.0'

# default model
MODEL_NAME = API_TITLE
MODEL_ID = MODEL_NAME.lower().replace(' ', '-')
DEFAULT_MODEL_PATH = 'assets/facenet.pb'
MODEL_LICENSE = 'MIT'

DEFAULT_IMAGE_SIZE = 160
DEFAULT_BATCH_SIZE = 2
DEFAULT_PREPROCESS_THREADS = 2
