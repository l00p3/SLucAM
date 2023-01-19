// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <SLucAM_image.h>
#include <SLucAM_geometry.h>
#include <SLucAM_initialization.h>
#include <SLucAM_measurement.h>
#include <SLucAM_state.h>
#include <SLucAM_dataset.h>
#include <SLucAM_visualization.h>
#include <SLucAM_keyframe.h>

using namespace std;



int main(int argc, char *argv[]) {

    // -----------------------------------------------------------------------------
    // Parse arguments
    // -----------------------------------------------------------------------------
    if(argc != 3) {
        std::cout << "ERROR!" << std::endl << "USAGE: SLucAM path_to_dataset features_type" << std::endl;
        return 1;
    }
    std::string dataset_name = argv[1];
    std::string features = argv[2];
    if(features != "orb" && features != "superpoint" && features != "lf_net") {
        std::cout << "The only available features are: orb, superpoint, lf_net" << std::endl;
            return 1;
    }

    // --- LIST OF AVAILABLE DATASETS ---
    //std::string dataset_name = "fr1_xyz";
    //std::string dataset_name = "fr2_xyz";
    //std::string dataset_name = "fr2_desk";
    //std::string dataset_name = "fr1_desk";
    //std::string dataset_name = "fr3_structure_texture_far";
    //std::string dataset_name = "fr3_structure_texture_near";
    //std::string dataset_name = "fr3_nostructure_texture_near";



    // -----------------------------------------------------------------------------
    // Create Environment and set variables
    // -----------------------------------------------------------------------------
    const bool verbose = true;
    const bool save_exploration = true;
    unsigned int step = 0;
    const unsigned int kernel_threshold_POSIT = 1000;
    const float inliers_threshold_POSIT = kernel_threshold_POSIT;

    std::string dataset_folder = "../data/datasets/" + dataset_name + "/";
    const std::string results_folder = "../results/" + dataset_name + \
                                        "_results_" + features + "/";

    // Determine if the dataset is slow or fast and set the density of keyframes
    unsigned int keyframes_density = 20;
    if(dataset_name == "fr1_desk")
        keyframes_density = 5;

    SLucAM::State state(keyframes_density);

    

    // -----------------------------------------------------------------------------
    // Load Dataset
    // -----------------------------------------------------------------------------
    SLucAM::FeatureExtractor feature_extractor = SLucAM::FeatureExtractor(false);
    cout << endl << "--- LOADING THE DATASET ---" << endl;
    bool loaded;
    if(features == "orb")
        loaded = SLucAM::load_TUM_dataset(dataset_folder, state, feature_extractor, keyframes_density, verbose);
    else if(features == "superpoint" || features == "lf_net")
        loaded = SLucAM::load_preextracted_TUM_dataset(dataset_folder, features+"/", state, keyframes_density, verbose);
    else {
        std::cout << "ERROR: invalid features type: " << features << std::endl;
        return 1;
    }
    if(!loaded) {
        cout << "ERROR: unable to load the specified dataset, check that it exists in the data/datasets/ folder" << endl;
        return 1;
    }
    cout << "--- DONE! ---" << endl << endl;



    // -----------------------------------------------------------------------------
    // Create Matcher
    // -----------------------------------------------------------------------------
    SLucAM::Matcher matcher(features);



    // -----------------------------------------------------------------------------
    // INITIALIZATION
    // -----------------------------------------------------------------------------
    cout << "--- INITIALIZATION ---" << endl;
    if(!state.initializeState(matcher, verbose)) {
        cout << "ERROR: unable to perform initialization" << endl;
        return 1;
    }
    if(save_exploration) {
        SLucAM::save_current_state(results_folder+"frame"+std::to_string(step)+"_", state);
        step++;
    }
    cout << "--- DONE! ---" << endl << endl;



    // -----------------------------------------------------------------------------
    // INTEGRATE NEW MEASUREMENT AND EXPAND MAP
    // -----------------------------------------------------------------------------
    cout << "--- ESPLORATION STARTED ---" << endl;
    while(state.reaminingMeasurements() != 0) {
        if(state.integrateNewMeasurement(matcher, \
                                    true, \
                                    kernel_threshold_POSIT, \
                                    inliers_threshold_POSIT, \
                                    verbose)) {
            if(save_exploration) {
                SLucAM::save_current_state(results_folder+"frame"+std::to_string(step)+"_", state);
                step++;
            }
        } else {
            std::cout << std::endl << "-> TRACKING LOOSE AT MEAS " << state.getLastMeasurement().getId() << std::endl;
            break;
        }
    }
    cout << "--- DONE! ---" << endl << endl;



    // -----------------------------------------------------------------------------
    // PERFORM FINAL BUNDLE ADJUSTMENT
    // -----------------------------------------------------------------------------
    state.performTotalBA(10, verbose);
    if(save_exploration) {
        SLucAM::save_current_state(results_folder+"frame"+std::to_string(step)+"_", state);
        std::cout << "Last state saved is: " << results_folder+"frame"+std::to_string(step)+"_*" << std::endl;
        step++;
    }



    // -----------------------------------------------------------------------------
    // SAVE RESULTS
    // -----------------------------------------------------------------------------
    SLucAM::save_TUM_results(dataset_folder, features, state);



    return 0;
}