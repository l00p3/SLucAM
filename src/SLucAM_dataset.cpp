//
// SLucAM_dataset.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_dataset.h>
#include <SLucAM_image.h>
#include <SLucAM_geometry.h>
#include <SLucAM_keyframe.h>
#include <filesystem>
#include <fstream>
#include <iostream>



// -----------------------------------------------------------------------------
// Implementation of functions to deal with my personal dataset format
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Function to load my personal dataset.
    * Inputs:
    *   dataset_folder: folder where to find all the images and
    *           the specification of the camera matrix in a file
    *           with name "camera.dat"
    *   state: state object where to store loaded infos
    */
    bool load_my_dataset(const std::string& dataset_folder, State& state, \
                        const cv::Ptr<cv::Feature2D>& detector, \
                        const unsigned int keyframe_density, \
                        const bool verbose) {

        // Initialization
        std::string current_filename, current_line;
        std::string camera_matrix_filename = dataset_folder + "camera.dat";
        std::string csv_filename = dataset_folder + "data.csv";
        cv::Mat current_img, K, distorsion_coefficients;
        bool K_loaded = false;
        std::vector<Measurement> measurements;

        // Load the camera matrix
        if(!load_camera_matrix(camera_matrix_filename, K, distorsion_coefficients))
            return false;

        // Open the csv file
        std::fstream csv_file;
        csv_file.open(csv_filename);
        if(csv_file.fail()) return false;
        
        // Load all measurements
        measurements.reserve(35);
        while(std::getline(csv_file, current_line)) {

            // Get the current filename
            std::stringstream ss_current_line_csv_file(current_line);
            ss_current_line_csv_file >> current_filename; 
            ss_current_line_csv_file >> current_filename;
            current_filename = dataset_folder+current_filename;

            // Load the measurement
            if(!load_image(current_filename, current_img))
                return false;
            
            // Detect keypoints
            std::vector<cv::KeyPoint> points;
            cv::Mat descriptors;
            detector->detectAndCompute(current_img, cv::Mat(), \
                                            points, descriptors);

            // Undistort keypoints
            std::vector<cv::KeyPoint> undistorted_points;
            undistort_keypoints(points, undistorted_points, \
                                distorsion_coefficients, K);

            // Create new measurement
            measurements.emplace_back(Measurement(undistorted_points, \
                                        descriptors));

            // Memorize the name of the image
            measurements.back().setImgName(current_filename);

        }
        measurements.shrink_to_fit();

        // Initialize the state
        state = State(K, distorsion_coefficients, measurements, \
                        measurements.size()+1, 10000, keyframe_density);

        if(verbose) {
            std::cout << "Loaded " << measurements.size() << " measurements" \
                << " with camera matrix:" << std::endl << K << std::endl \
                << "and distorsion parameters: " << std::endl \
                << distorsion_coefficients << std::endl;
        }
        
        return true;
            
    }


    /*
    * Load the camera matrix from the "camera.dat" file in my dataset.
    * It returns false in case of errors.
    */
    bool load_camera_matrix(const std::string& filename, cv::Mat& K, \
                            cv::Mat& distorsion_coefficients) {

        K = cv::Mat::zeros(3,3,CV_32F);
        distorsion_coefficients = cv::Mat::zeros(1,5,CV_32F);

        std::fstream camera_file;
        camera_file.open(filename);
        if(camera_file.fail()) return false;
        camera_file >> \
            K.at<float>(0,0) >> K.at<float>(0,1) >> K.at<float>(0,2) >> \
            K.at<float>(1,0) >> K.at<float>(1,1) >> K.at<float>(1,2) >> \
            K.at<float>(2,0) >> K.at<float>(2,1) >> K.at<float>(2,2) >> \
            distorsion_coefficients.at<float>(0,0) >> \
            distorsion_coefficients.at<float>(0,1) >> \
            distorsion_coefficients.at<float>(0,2) >> \
            distorsion_coefficients.at<float>(0,3) >> \
            distorsion_coefficients.at<float>(0,4);
        camera_file.close();

        return true;
    }

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of functions to deal with the TUM Dataset
// -----------------------------------------------------------------------------
namespace SLucAM {

    bool load_TUM_dataset(const std::string& dataset_folder, State& state, \
                            FeatureExtractor& feature_extractor, \
                            const unsigned int keyframe_density, \
                            const bool verbose) {

        // Initialization
        std::string camera_matrix_filename = dataset_folder + "camera_parameters.txt";
        std::string imgs_names_filename = dataset_folder + "rgb.txt";
        std::string current_line, current_img_filename;
        cv::Mat K, distorsion_coefficients, current_img;
        std::vector<Measurement> measurements;

        // Load the camera matrix
        if(!load_TUM_camera_matrix(camera_matrix_filename, K, distorsion_coefficients))
            return false;
        
        // Open the file containing the ordered names of the images
        std::fstream imgs_names_file;
        imgs_names_file.open(imgs_names_filename);
        if(imgs_names_file.fail()) return false;

        // Ignore the first three lines
        std::getline(imgs_names_file, current_line);
        std::getline(imgs_names_file, current_line);
        std::getline(imgs_names_file, current_line);

        // Load all measurements
        int i = 0;
        measurements.reserve(4000);
        while(std::getline(imgs_names_file, current_line)) {

            //if(i==300) break;
            
            // Get the current filename
            std::stringstream ss_current_line_csv_file(current_line);
            ss_current_line_csv_file >> current_img_filename; 
            ss_current_line_csv_file >> current_img_filename;
            current_img_filename = dataset_folder+current_img_filename;

            // Load the measurement (undistorted)
            if(!load_image(current_img_filename, current_img))
                return false;
            
            // Detect keypoints
            std::vector<cv::KeyPoint> points;
            cv::Mat descriptors;
            feature_extractor.extract_features(current_img, points, descriptors);
                        
            // Undistort keypoints
            std::vector<cv::KeyPoint> undistorted_points;
            undistort_keypoints(points, undistorted_points, \
                                distorsion_coefficients, K);

            // Create new measurement
            measurements.emplace_back(Measurement(undistorted_points, 
                                        descriptors));

            // Memorize the name of the image
            measurements.back().setImgName(current_img_filename);

            ++i;

        }
        measurements.shrink_to_fit();

        // Close the file
        imgs_names_file.close();

        // Initialize the state
        state = State(K, distorsion_coefficients, measurements, \
                        measurements.size()+1, 10000, keyframe_density);

        if(verbose) {
            std::cout << "Loaded " << measurements.size() << " measurements" \
                << " with camera matrix:" << std::endl << K << std::endl \
                << "and distorsion parameters: " << std::endl \
                << distorsion_coefficients << std::endl;
        }

        return true;

    }


    bool load_preextracted_TUM_dataset(const std::string& dataset_folder, \
                            const std::string& features_folder, State& state, \
                            const unsigned int keyframe_density, \
                            const bool verbose) {
        
        // Initialization
        const std::string camera_matrix_filename = dataset_folder + "camera_parameters.txt";
        const std::string imgs_names_filename = dataset_folder + "rgb.txt";
        const std::string features_folder_path = dataset_folder + features_folder;
        std::string current_line, current_img_filename, current_filename;
        float x,y,d;
        cv::Mat K, distorsion_coefficients;
        std::vector<Measurement> measurements;

        // Load the camera matrix
        if(!load_TUM_camera_matrix(camera_matrix_filename, K, distorsion_coefficients))
            return false;

        // Open the file containing the ordered names of the images
        std::fstream imgs_names_file;
        imgs_names_file.open(imgs_names_filename);
        if(imgs_names_file.fail()) return false;

        // Ignore the first three lines
        std::getline(imgs_names_file, current_line);
        std::getline(imgs_names_file, current_line);
        std::getline(imgs_names_file, current_line);

        // Load all measurements
        int i = 0;
        measurements.reserve(4000);
        while(std::getline(imgs_names_file, current_line)) {

            //if(i==300) break;
            
            // Get the current filename
            std::stringstream ss_current_line_csv_file(current_line);
            ss_current_line_csv_file >> current_filename; 
            ss_current_line_csv_file >> current_filename;
            current_img_filename = dataset_folder+current_filename;
            current_filename = features_folder_path+current_filename.substr(4);
            current_filename = current_filename.substr(0,current_filename.size()-3) + "dat";

            // Open the current file
            std::fstream current_file;
            current_file.open(current_filename);
            if(current_file.fail()) return false;

            // Load all points and associated descriptors
            std::vector<cv::KeyPoint> points; points.reserve(1000);
            std::vector<std::vector<float>> desc; desc.reserve(1000);
            while(std::getline(current_file, current_line)) {
                
                // Get the stringstream
                std::stringstream ss_current_line_feats(current_line);

                // Read the point
                x, y;
                ss_current_line_feats >> x >> y;
                points.emplace_back(cv::KeyPoint(cv::Point2f(x,y),1.0));

                // Read the descriptor
                std::vector<float> current_desc; current_desc.reserve(1000);
                while(ss_current_line_feats >> d) {
                    current_desc.emplace_back(d);
                }
                current_desc.shrink_to_fit();
                desc.emplace_back(current_desc);

            }
            points.shrink_to_fit(); desc.shrink_to_fit();

            // Convert the descriptors in cv::Mat
            const unsigned int n_points = desc.size();
            const unsigned int length_desc = desc.back().size();
            cv::Mat descriptors = cv::Mat::zeros(n_points, length_desc, CV_32F);
            unsigned int c,r = 0;
            for(const auto& row: desc) {
                c = 0;
                for(const auto& col: row) {
                    descriptors.at<float>(r,c) = col;
                    ++c;
                }
                ++r;
            }

            // Create new measurement
            measurements.emplace_back(Measurement(points, descriptors));

            // Memorize the name of the image
            measurements.back().setImgName(current_img_filename);

            ++i;

        }
        measurements.shrink_to_fit();

        // Close the file
        imgs_names_file.close();

        // Initialize the state
        state = State(K, distorsion_coefficients, measurements, \
                        measurements.size()+1, 10000, keyframe_density);

        if(verbose) {
            std::cout << "Loaded " << measurements.size() << " measurements" \
                << " with camera matrix:" << std::endl << K << std::endl \
                << "and distorsion parameters: " << std::endl \
                << distorsion_coefficients << std::endl;
        }

        return true;
        
    }


    bool load_TUM_camera_matrix(const std::string& filename, cv::Mat& K, \
                                cv::Mat& distorsion_coefficients) {
        
        K = cv::Mat::zeros(3,3,CV_32F);
        distorsion_coefficients = cv::Mat::zeros(1,5,CV_32F);

        std::fstream camera_file;
        camera_file.open(filename);
        if(camera_file.fail()) return false;
        camera_file >> \
            K.at<float>(0,0) >> K.at<float>(0,1) >> K.at<float>(0,2) >> \
            K.at<float>(1,0) >> K.at<float>(1,1) >> K.at<float>(1,2) >> \
            K.at<float>(2,0) >> K.at<float>(2,1) >> K.at<float>(2,2) >> \
            distorsion_coefficients.at<float>(0,0) >> \
            distorsion_coefficients.at<float>(0,1) >> \
            distorsion_coefficients.at<float>(0,2) >> \
            distorsion_coefficients.at<float>(0,3) >> \
            distorsion_coefficients.at<float>(0,4);
        camera_file.close();

        return true;

    }



    bool save_TUM_results(const std::string& dataset_folder, const std::string& features, \
                            const State& state) {

        // Initialization
        std::string results_filename = dataset_folder + "SLucAM_results_" + features + ".txt";
        std::string current_line;
        const std::vector<Keyframe>& keyframes = state.getKeyframes();
        const std::vector<cv::Mat>& poses = state.getPoses();
        const std::vector<Measurement>& measurements = state.getMeasurements();
        std::string delimiter = "/";
        int start, end;

        // Open the file where to save the results
        std::ofstream results_file;
        results_file.open(results_filename);
        if(results_file.fail()) return false;
        results_file << std::fixed;

        // Save each keyframe's pose in the state, in the TUM format
        for(const Keyframe& kf : keyframes) {
            
            // Get the pose of the current keyframe
            const cv::Mat current_pose = invert_transformation_matrix(poses[kf.getPoseIdx()]);
            cv::Mat current_quat;
            matrix_to_quaternion(current_pose.rowRange(0,3).colRange(0,3), \
                    current_quat);
            
            // Get the timestamp of the current keyframe from the image from which it is
            // extracted
            const std::string current_img_name = measurements[kf.getMeasIdx()].getImgName();    
            start = 0;
            end = current_img_name.find(delimiter);
            while (end != -1) {
                start = end + delimiter.size();
                end = current_img_name.find(delimiter, start);
            }
            const double current_timestamp = std::stod(current_img_name.substr(start, end - start));

            results_file << std::setprecision(6) \
                << current_timestamp << " " \
                << std::setprecision(7) \
                << current_pose.at<float>(0,3) << " " \
                << current_pose.at<float>(1,3) << " " \
                << current_pose.at<float>(2,3) << " " \
                << current_quat.at<float>(0,0) << " " \
                << current_quat.at<float>(1,0) << " " \
                << current_quat.at<float>(2,0) << " " \
                << current_quat.at<float>(3,0) << std::endl;
        }

        // Close the results file
        results_file.close();

        return true;

    }

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of functions to save and load general infos on files
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Function that save the current state in a folder by saving all
    * the poses and landmarks, the filename of the last seen image
    * and the list of landmarks seen from the last keyframe/pose.
    */
    bool save_current_state(const std::string& folder, \
                            const State& state) {
        
        // Initialization
        const Keyframe& last_keyframe = state.getKeyframes().back();
        const Measurement& last_measure = state.getLastMeasurement();
        const std::string& last_image = last_measure.getImgName();
        const std::vector<cv::KeyPoint>& last_measure_points = last_measure.getPoints();
        const std::string last_image_filename = folder + "SLucAM_image_name.dat";

        // Save the filename of the last seen image
        std::ofstream f_img;
        f_img.open(last_image_filename);
        if(f_img.fail()) return false;
        f_img << last_image;
        f_img.close();
        
        // Save all the keyframe's poses and the last pose
        if(!save_poses(folder, state.getKeyframes(), state.getPoses())) return false;

        // Save all landmarks
        if(!save_landmarks(folder, state.getKeypoints())) return false;

        // Save the edges last measurement <-> landmarks
        if(!save_edges(folder, last_measure)) return false;

        // Save the points on the last image
        if(!save_keypoints(folder, last_measure_points)) return false;

        return true;

    }


    /*
    * Function that save in a file all the predicted poses with the 
    * format: tx ty tz r11 r12 r13 ... r33 (where rij is the element
    * in the position <i,j> of the rotation part of the pose).
    * In particular it saves all the poses of the spawned keyframes
    * and the last pose (the pose of the last integrated measurement)
    */  
    bool save_poses(const std::string& folder, \
                    const std::vector<Keyframe>& keyframes, \
                    const std::vector<cv::Mat>& poses) {
        
        // Initialization
        const std::string filename = folder + "SLucAM_poses.dat";
        const unsigned int n_keyframes = keyframes.size();

        // Open the file
        std::ofstream f;
        f.open(filename);
        if(f.fail()) return false;

        // Write the header
        f << "PREDICTED POSES" << std::endl;
        f << "tx\tty\ttz\tr11\tr12\tr13\tr21\tr22\tr23\tr31\tr32\tr33\t";

        // Write all the keyframes' poses
        for(unsigned int i=0; i<n_keyframes; ++i) {
            const cv::Mat& p = poses[keyframes[i].getPoseIdx()];
            f << std::endl \
                << p.at<float>(0,3) << "\t" << p.at<float>(1,3) << "\t" << p.at<float>(2,3) << "\t" \
                << p.at<float>(0,0) << "\t" << p.at<float>(0,1) << "\t" << p.at<float>(0,2) << "\t" \
                << p.at<float>(1,0) << "\t" << p.at<float>(1,1) << "\t" << p.at<float>(1,2) << "\t" \
                << p.at<float>(2,0) << "\t" << p.at<float>(2,1) << "\t" << p.at<float>(2,2);
                
        }

        // Write the last pose (only if it is not the pose of the last 
        // keyframe)
        if(keyframes.back().getPoseIdx() != poses.size()-1) {
            const cv::Mat& p = poses.back();
            f << std::endl \
                << p.at<float>(0,3) << "\t" << p.at<float>(1,3) << "\t" << p.at<float>(2,3) << "\t" \
                << p.at<float>(0,0) << "\t" << p.at<float>(0,1) << "\t" << p.at<float>(0,2) << "\t" \
                << p.at<float>(1,0) << "\t" << p.at<float>(1,1) << "\t" << p.at<float>(1,2) << "\t" \
                << p.at<float>(2,0) << "\t" << p.at<float>(2,1) << "\t" << p.at<float>(2,2);
        }

        // Close the file
        f.close();

        return true;

    }


    /*
    * Function that save in a file all the predicted 3D points.
    */  
    bool save_landmarks(const std::string& folder, \
                        const std::vector<Keypoint>& keypoints) {
        
        // Initialization
        const std::string filename = folder + "SLucAM_landmarks.dat";
        unsigned int n_keyopints = keypoints.size();

        // Open the file
        std::ofstream f;
        f.open(filename);
        if(f.fail()) return false;

        // Write the header
        f << "3D PREDICTED POINTS" << std::endl;
        f << "x\ty\tz";

        // Write all the landmarks
        for(unsigned int i=0; i<n_keyopints; ++i) {
            const cv::Point3f& l = keypoints[i].getPosition();
            f << std::endl << l.x << "\t" << l.y << "\t" << l.z;
        }

        // Close the file
        f.close();

        return true;

    }


    /*
    * Function that saves in a file the list of landmarks indices seen
    * from the given keyframe.
    */
    bool save_edges(const std::string& folder, \
                    const Measurement& measurement) {
        
        // Initialization
        const std::string filename = folder + "SLucAM_edges.dat";
        std::vector<std::pair<unsigned int, unsigned int>> points_associations;
        measurement.getPointsAssociations(points_associations);
        const unsigned int n_edges = points_associations.size();

        // Open the file
        std::ofstream f;
        f.open(filename);
        if(f.fail()) return false;

        // Write the header
        f << "LIST OF SEEN LANDMARKS" << std::endl;
        f << "landmark_idx";

        // Write all the landmarks
        for(unsigned int i=0; i<n_edges; ++i) {
            f << std::endl << points_associations[i].second;
        }

        // Close the file
        f.close();

        return true;

    }


    /*
    * Function that saves in a file the list of points positions seen
    * on a measurement.
    */
    bool save_keypoints(const std::string& folder, \
                        const std::vector<cv::KeyPoint>& points) {
        
        // Initialization
        const std::string filename = folder + "SLucAM_img_points.dat";
        unsigned int n_points = points.size();

        // Open the file
        std::ofstream f;
        f.open(filename);
        if(f.fail()) return false;

        // Write the header
        f << "2D POINTS ON IMAGE" << std::endl;
        f << "x\ty";

        // Write all the landmarks
        for(unsigned int i=0; i<n_points; ++i) {
            const cv::KeyPoint& p = points[i];
            f << std::endl << p.pt.x << "\t" << p.pt.y;
        }

        // Close the file
        f.close();

        return true;

    }

} // namespace SLucAM