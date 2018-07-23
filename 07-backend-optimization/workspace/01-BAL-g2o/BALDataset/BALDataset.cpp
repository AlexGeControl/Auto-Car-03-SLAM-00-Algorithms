#include "BALDataset.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include <cmath>
#include <limits>

#include "rotation.h"

using namespace std;

// states for BAL dataset parsing
enum BALState {
    NUMS,
    OBSERVATIONS,
    CAMERAS,
    POINTS,
    DONE,
    EXIT
};

BALDataset::BALDataset(const std::string &filename) {
    // open dataset:
    ifstream dataset(filename.c_str(), ifstream::in);

    // whether the file could be opened:
    if (!dataset) {
        // failed to open: abort
        cerr << "[BALDataset]: Unable to load dataset " << filename << endl;
        exit(1);
    } else {
        cout << "[BALDataset]: Loading dataset " << filename << "..." << endl; 
    }

    // parse BAL dataset:
    BALState state = NUMS;

    double camera_params[DIM_CAMERA_INIT_PARAMS];
    double quaternion[4];
    double point_params[DIM_POINT_INIT_PARAMS];

    do {
        switch (state) {
            // stage 1: parse dimensions
            case NUMS: 
            {
                if (!(dataset >> num_cameras >> num_points >> num_observations)) {
                    cerr << "[BALDataset]: Parse dataset dimensions failed." << endl;
                    exit(1);
                }
                else {
                    state = OBSERVATIONS;
                }

                break;
            }
            // stage 2: parse observations
            case OBSERVATIONS: {
                Observation observation;
                double u, v;

                if (!(dataset >> observation.camera_index >> observation.point_index >> u >> v)) {
                    cerr << "[BALDataset]: Parse observation failed." << endl;
                    exit(1);
                }
                else {
                    observation.measurement = Eigen::Vector2d(u, v);
                    observations.push_back(observation);

                    if (observations.size() == num_observations) {
                        state = CAMERAS;
                    }
                }

                break;
            }
            // stage 3: parse camera init
            case CAMERAS: {
                for (int i = 0; i < DIM_CAMERA_INIT_PARAMS; ++i) {
                    if (!(dataset >> camera_params[i])) {
                        cerr << "[BALDataset]: Parse camera params failed." << endl;
                        exit(1);                        
                    }
                }

                Eigen::VectorXd camera(9);
                camera << \
                    // a. pose, rotation:
                    camera_params[0], camera_params[1], camera_params[2], \
                    // b. pose, translation:
                    camera_params[3], camera_params[4], camera_params[5], \
                    // c. intrinsic:
                    camera_params[6], \
                    // d. distortion:
                    camera_params[7], camera_params[8];
                cameras.push_back(camera);

                if (cameras.size() == num_cameras) {
                    state = POINTS;
                }

                break;
            }
            // stage 4: parse point init
            case POINTS: {
                for (int i = 0; i < DIM_POINT_INIT_PARAMS; ++i) {
                    if (!(dataset >> point_params[i])) {
                        cerr << "[BALDataset]: Parse point params failed." << endl;
                        exit(1);                        
                    }
                }

                Eigen::Vector3d point;
                point.x() = point_params[0];
                point.y() = point_params[1];
                point.z() = point_params[2];
                points.push_back(point);

                if (points.size() == num_points) {
                    state = DONE;
                }                
                     
                break;
            }
            // stage final: done
            case DONE: {
                cout << "[BALDataset]: num cameras / num points / num observations -- " \
                     << num_cameras << ", " << num_points << ", " << num_observations << endl;
                state = EXIT; 
                break;
            }
            case EXIT: {
                break;
            }
            default:
                break;
        }
    } while (EXIT != state);

    dataset.close();
}

/**
    get camera center in world frame.

    @param camera the camera instance in BAL format.
    @param result the output camera center in world frame.
*/
void BALDataset::get_camera_center_in_world_frame(const Eigen::VectorXd &camera, double result[3]) {
    // parse angle axis:
    double angle_axis[3] = {
        -camera(0), -camera(1), -camera(2)
    };
    // parse point:
    double point[3] = {
        -camera(3), -camera(4), -camera(5)
    };

    // p_camera_center_world = -R_prime * t
    utils::rotation::rotate_point_around_angle_axis(
        angle_axis, point, result
    );
}

/**
    write camera poses and structures to PLY file.

    @param filename the output PLY filename.
*/
void BALDataset::write_to_ply(const std::string& filename) const {
    std::ofstream ply(filename.c_str());

    // header:
    ply << "ply"
        << '\n' << "format ascii 1.0"
        << '\n' << "element vertex " << num_cameras + num_points
        << '\n' << "property float x"
        << '\n' << "property float y"
        << '\n' << "property float z"
        << '\n' << "property uchar red"
        << '\n' << "property uchar green"
        << '\n' << "property uchar blue"
        << '\n' << "end_header" << std::endl;

    // export extrinsic data (i.e. camera centers) as green points.
    double camera_center[3];
    for(int i = 0; i < cameras.size(); ++i){
        // camera center statisfy: 0_center_camera = R*p_center_world + t
        // so p_center_world = -R_prime*t
        get_camera_center_in_world_frame(cameras[i], camera_center);

        ply << camera_center[0] << ' ' << camera_center[1] << ' ' << camera_center[2] << ' ';
        ply << "0 255 0" << endl;
    }

    // export the structure (i.e. 3D Points) as white points.
    for (int i = 0; i < points.size(); ++i) {
        const Eigen::Vector3d &point = points[i];

        ply << point.x() << ' ' << point.y() << ' ' << point.z() << ' ';
        ply << "255 255 255" << endl;
    }

    ply.close();
}