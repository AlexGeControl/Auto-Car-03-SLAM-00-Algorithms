// include guard:
#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

class BALDataset {
public:
    static const int DIM_CAMERA_INIT_PARAMS = 9;
    static const int DIM_POINT_INIT_PARAMS = 3;

    // observations:
    struct Observation {
        int camera_index;
        int point_index;
        Eigen::Vector2d measurement; 
    };

    // constructor:
    BALDataset(const std::string &filename);

    // accessors:
    int get_num_cameras() const {return num_cameras;}
    int get_num_points() const {return num_points;}
    int get_num_observations() const {return num_observations;}    

    const std::vector<Eigen::VectorXd> &get_cameras() const {return cameras;}
    const std::vector<Eigen::Vector3d> &get_points() const {return points;}
    const std::vector<Observation> &get_observations() const {return observations;}

    // convertors:
    static void get_camera_center_in_world_frame(const Eigen::VectorXd &camera, double result[3]);

    // IO:  
    void write_to_ply(const std::string& filename) const;
private:
    // dimensions:
    int num_cameras;
    int num_points;
    int num_observations;

    // observations:
    std::vector<Observation> observations;

    // cameras:
    std::vector<Eigen::VectorXd> cameras;

    // points:
    std::vector<Eigen::Vector3d> points;
};