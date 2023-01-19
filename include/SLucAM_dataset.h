//
// SLucAM_dataset.h
//
// In this module we have all the function to deal with dataset of
// different format.
//


#ifndef SLUCAM_DATASET_H
#define SLUCAM_DATASET_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <string>
#include <opencv2/features2d.hpp>
#include <SLucAM_state.h>
#include <SLucAM_image.h>
#include <SLucAM_keyframe.h>
#include <SLucAM_keypoint.h>



// -----------------------------------------------------------------------------
// Functions to deal with my personal dataset format
// -----------------------------------------------------------------------------
namespace SLucAM {
    bool load_my_dataset(const std::string& dataset_folder, State& state, \
                            const cv::Ptr<cv::Feature2D>& detector, \
                            const unsigned int keyframe_density, \
                            const bool verbose=false);
    bool load_camera_matrix(const std::string& filename, cv::Mat& K, \
                            cv::Mat& distorsion_coefficients);
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Functions to deal with the TUM Dataset
// -----------------------------------------------------------------------------
namespace SLucAM {
    bool load_TUM_dataset(const std::string& dataset_folder, State& state, \
                            FeatureExtractor& feature_extractor, \
                            const unsigned int keyframe_density, \
                            const bool verbose=false);
    bool load_preextracted_TUM_dataset(const std::string& dataset_folder, \
                            const std::string& features_folder, State& state, \
                            const unsigned int keyframe_density, \
                            const bool verbose=false);
    bool load_TUM_camera_matrix(const std::string& filename, cv::Mat& K, \
                                cv::Mat& distorsion_coefficients);
    bool save_TUM_results(const std::string& dataset_folder, const std::string& features, \
                            const State& state);
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Functions to save and load general infos on files
// -----------------------------------------------------------------------------
namespace SLucAM {
    bool save_current_state(const std::string& folder, \
                            const State& state);
    bool save_poses(const std::string& folder, \
                    const std::vector<Keyframe>& keyframes, \
                    const std::vector<cv::Mat>& poses);
    bool save_landmarks(const std::string& folder, \
                        const std::vector<Keypoint>& keypoints);
    bool save_edges(const std::string& folder, \
                    const Measurement& measurement);
    bool save_keypoints(const std::string& folder, \
                        const std::vector<cv::KeyPoint>& points);
} // namespace SLucAM



#endif // SLUCAM_STATE_H