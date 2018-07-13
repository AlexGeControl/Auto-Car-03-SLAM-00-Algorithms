#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;

// default input path:
string p3d_file = "./p3d.txt";
string p2d_file = "./p2d.txt";

int main(int argc, char **argv) {
    // intrinsic params:
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

    // memory buffer for p3d and p2d
    double v[3];
    int D, d;

    // perspectives and points:
    vector<cv::Point3d> pts3d;
    vector<cv::Point2d> pts2d;
    
    // points:
    ifstream p3d_filestream(p3d_file);
    D = 3; d = 0;
    while (p3d_filestream >> v[d]) {
        // update index:
        d = (d + 1) % D;

        // update point:
        if (0 == d) {
            pts3d.push_back(cv::Point3d(v[0], v[1], v[2]));
        }
    }
    // perspectives:
    ifstream p2d_filestream(p2d_file);
    D = 2; d = 0;
    while (p2d_filestream >> v[d]) {
        // update index:
        d = (d + 1) % D;

        // update point:
        if (0 == d) {
            pts2d.push_back(cv::Point2d(v[0], v[1]));
        }
    }   
    assert(pts3d.size() == pts2d.size());

    // total number of observations:
    const int N = pts3d.size();

    // use OpenCV to generate initial estimation:
    cv::Mat q, t;
    cv::solvePnP(pts3d, pts2d, K, cv::Mat(), q, t, false);
    cv::Mat R;
    cv::Rodrigues(q, R);

    // display initial estimation:
    cout << "R_init = " << endl;
    cout << R << endl;
    cout << "t_init = " << endl;
    cout << t << endl;

    // build optimization graph:
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolver;
    BlockSolver::LinearSolverType *linear_solver = new g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>();
    BlockSolver *block_solver = new BlockSolver(linear_solver);
    g2o::OptimizationAlgorithmLevenberg *algorithm = new g2o::OptimizationAlgorithmLevenberg(block_solver);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    // set vertices:
    // a. Pose:
    Eigen::Matrix3d R_init;
    R_init << \
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
    Eigen::Vector3d t_init;
    t_init << t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0);

    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(
        g2o::SE3Quat(
            R_init,
            t_init
        )
    );
    optimizer.addVertex(pose);

    // b. points:
    for (int i = 0; i < N; ++i) {
        const auto &p3d = pts3d[i];

        Eigen::Vector3d point_init;
        point_init << p3d.x, p3d.y, p3d.z;

        g2o::VertexSBAPointXYZ *point = new g2o::VertexSBAPointXYZ();
        point->setId(i + 1);
        point->setEstimate(point_init);
        point->setMarginalized(true);
        point->setFixed(false);
        optimizer.addVertex(point);
    } 

    // camera:
    g2o::CameraParameters *camera = new g2o::CameraParameters(
        fx,
        Eigen::Vector2d(cx, cy),
        0.0
    );
    camera->setId(0);
    optimizer.addParameter(camera);

    // set edges:
    for (int i = 0; i < N; ++i) {
        const auto &p2d = pts2d[i];

        g2o::EdgeProjectXYZ2UV *pixel = new g2o::EdgeProjectXYZ2UV();
        pixel->setId(i);
        pixel->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(i + 1)));
        pixel->setVertex(1, pose);
        pixel->setMeasurement(Eigen::Vector2d(p2d.x, p2d.y));
        pixel->setParameterId(0, 0);
        pixel->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(pixel);
    }

    // optimize:
    const int MAX_ITER = 100;
    chrono::steady_clock::time_point start_time = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(MAX_ITER);
    chrono::steady_clock::time_point end_time = chrono::steady_clock::now();

    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(end_time - start_time);
    cout << "g2o solver costs " << time_used.count() << " seconds." << endl;

    // display result:
    cout << "[Optimized Pose]: " << endl << Eigen::Isometry3d(pose->estimate()).matrix() << endl;

    return 0;
}
