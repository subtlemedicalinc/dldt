"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.extractor import FrontExtractorOp
from mo.ops.softmax import Softmax


class SoftmaxComponentFrontExtractor(FrontExtractorOp):
    op = 'softmaxcomponent'
    enabled = True

    @staticmethod
    def extract(node):
        return SoftmaxFrontExtractor.extract(node)


class SoftmaxFrontExtractor(FrontExtractorOp):
    op = 'softmax'
    enabled = True

    @staticmethod
    def extract(node):
        Softmax.update_node_stat(node, {'infer': copy_shape_infer})
        return __class__.enabled
