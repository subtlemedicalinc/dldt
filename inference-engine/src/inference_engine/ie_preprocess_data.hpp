// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "ie_blob.h"
#include "ie_input_info.hpp"
#include "ie_profiling.hpp"

namespace InferenceEngine {

/**
 * @brief This class stores pre-process information for exact input
 */
class INFERENCE_ENGINE_API_CLASS(PreProcessData) {
    /**
     * @brief ROI blob.
     */
    Blob::Ptr _roiBlob = nullptr;
    Blob::Ptr _tmp1 = nullptr;
    Blob::Ptr _tmp2 = nullptr;

    InferenceEngine::ProfilingTask perf_resize {"Resize"};
    InferenceEngine::ProfilingTask perf_reorder_before {"Reorder before"};
    InferenceEngine::ProfilingTask perf_reorder_after {"Reorder after"};
    InferenceEngine::ProfilingTask perf_preprocessing {"Preprocessing"};

public:
    /**
     * @brief Sets ROI blob to be resized and placed to the default input blob during pre-processing.
     * @param blob ROI blob.
     */
    void setRoiBlob(const Blob::Ptr &blob);

    /**
     * @brief Gets pointer to the ROI blob used for a given input.
     * @return Blob pointer.
     */
    Blob::Ptr getRoiBlob() const;

    /**
     * @brief Executes input pre-processing with a given resize algorithm.
     * @param outBlob pre-processed output blob to be used for inference.
     * @param algorithm resize algorithm.
     */
    void execute(Blob::Ptr &outBlob, const ResizeAlgorithm &algorithm);
};

}  // namespace InferenceEngine
