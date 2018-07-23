#include <string>
#include <chrono>

#include "BALDataset.h"
#include "BALClass.h"

#include "g2o/stuff/sampler.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"

using namespace std;

/**
    build optimizer for BAL.

    @param camera the camera instance in BAL format.
    @param result the output camera center in world frame.
*/ 
void build_optimizer(g2o::SparseOptimizer &optimizer) {
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9,3> > BALBlockSolver;

    // sparse solver:
    BALBlockSolver::LinearSolverType *linear_solver = new g2o::LinearSolverCholmod<BALBlockSolver::PoseMatrixType>();
    (dynamic_cast<g2o::LinearSolverCholmod<BALBlockSolver::PoseMatrixType>*>(linear_solver))->setBlockOrdering(true);
    // block solver:
    BALBlockSolver *block_solver = new BALBlockSolver(linear_solver);
    g2o::OptimizationAlgorithmLevenberg *algorithm = new g2o::OptimizationAlgorithmLevenberg(block_solver);
    // optimizer:
    optimizer.setAlgorithm(algorithm);
}

/**
    build optimization graph for BAL.

    @param camera the camera instance in BAL format.
    @param result the output camera center in world frame.
*/ 
void build_optimization_graph(const BALDataset &dataset, g2o::SparseOptimizer* optimizer)
{
    // add camera vertices:
    const std::vector<Eigen::VectorXd> &cameras = dataset.get_cameras();
    for(int i = 0; i < cameras.size(); ++i)
    {
        const Eigen::VectorXd &camera = cameras[i];
        VertexCameraBAL *vertex_camera = new VertexCameraBAL();

        CameraBAL camera_bal;
        // camera pose:
        double quaternion[4];
        utils::rotation::angle_axis_to_quaternion(camera.data(), quaternion);
        camera_bal.T = g2o::SE3Quat(
            Eigen::Quaternion<double>(quaternion),
            Eigen::Vector3d(camera(3), camera(4), camera(5))
        );
        // intrinsic:
        camera_bal.f = camera(6);
        // radial distortion:
        camera_bal.k1 = camera(7);
        camera_bal.k2 = camera(8);  

        vertex_camera->setEstimate(camera_bal);
        vertex_camera->setId(i);
  
        optimizer->addVertex(vertex_camera);
    }

    // add point vertices:
    const std::vector<Eigen::Vector3d> &points = dataset.get_points();
    const int point_id_base = cameras.size();
    for(int i = 0; i < points.size(); ++i)
    {
        const Eigen::Vector3d &point = points[i];
        VertexPointBAL *vertex_point = new VertexPointBAL();
        vertex_point->setEstimate(point);
        vertex_point->setId(point_id_base + i); 

        vertex_point->setMarginalized(true);
        optimizer->addVertex(vertex_point);
    }

    // add observation edges:
    const std::vector<BALDataset::Observation> &observations = dataset.get_observations();
    for(int i = 0; i < observations.size(); ++i)
    {
        const BALDataset::Observation &observation = observations[i];

        EdgeObservationBAL *edge_observation = new EdgeObservationBAL();

        // get id for camera and point:
        const int camera_id = observation.camera_index; 
        const int point_id = observation.point_index + point_id_base; 

        // set the vertex by the ids for an edge observation
        edge_observation->setVertex(
            0, dynamic_cast<VertexCameraBAL*>(optimizer->vertex(camera_id))
        );
        edge_observation->setVertex(
            1, dynamic_cast<VertexPointBAL*>(optimizer->vertex(point_id))
        );
        // information matrix:
        edge_observation->setInformation(
            Eigen::Matrix2d::Identity()
        );
        // loss function:
        g2o::RobustKernelHuber *robust_kernel = new g2o::RobustKernelHuber;
        robust_kernel->setDelta(1.0);
        edge_observation->setRobustKernel(robust_kernel);
        // measurement:
        edge_observation->setMeasurement(
            observation.measurement
        );
        optimizer->addEdge(edge_observation) ;
    }
}

/**
    solve BAL using g2o.

    @param camera the camera instance in BAL format.
    @param result the output camera center in world frame.
*/ 
void solve(g2o::SparseOptimizer &optimizer, int MAX_ITER = 100) {
    chrono::steady_clock::time_point start_time = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(MAX_ITER);
    chrono::steady_clock::time_point end_time = chrono::steady_clock::now();

    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(end_time - start_time);
    cout << "[BAL Problem Solved]: uses " << time_used.count() << " seconds." << endl;
}

/**
    write camera poses and structures to PLY file.

    @param filename the output PLY filename.
*/
void write_to_ply(
    const g2o::SparseOptimizer *optimizer, 
    const int NUM_CAMERAS, const int NUM_POINTS, 
    const std::string& filename
) {
    std::ofstream ply(filename.c_str());

    // header:
    ply << "ply"
        << '\n' << "format ascii 1.0"
        << '\n' << "element vertex " << NUM_CAMERAS + NUM_POINTS
        << '\n' << "property float x"
        << '\n' << "property float y"
        << '\n' << "property float z"
        << '\n' << "property uchar red"
        << '\n' << "property uchar green"
        << '\n' << "property uchar blue"
        << '\n' << "end_header" << std::endl;

    // export extrinsic data (i.e. camera centers) as green points.
    for(int i = 0; i < NUM_CAMERAS; ++i){
        const VertexCameraBAL* vertex_camera = dynamic_cast<const VertexCameraBAL*>(optimizer->vertex(i));
        const CameraBAL &camera = vertex_camera->estimate();
        // camera center statisfy: 0_center_camera = R*p_center_world + t
        // so p_center_world = -R_prime*t
        const Eigen::Matrix3d R = camera.T.rotation().toRotationMatrix();
        const Eigen::Vector3d t = camera.T.translation();
        const Eigen::Vector3d c = -R.transpose()*t;
        const double *camera_center = c.data();

        ply << camera_center[0] << ' ' << camera_center[1] << ' ' << camera_center[2] << ' ';
        ply << "0 255 0" << endl;
    }

    // export the structure (i.e. 3D Points) as white points.
    const int point_id_base = NUM_CAMERAS;
    for (int i = 0; i < NUM_POINTS; ++i) {
        const VertexPointBAL* vertex_point = dynamic_cast<const VertexPointBAL*>(optimizer->vertex(point_id_base + i));
        const Eigen::Vector3d &point = vertex_point->estimate();

        ply << point.x() << ' ' << point.y() << ' ' << point.z() << ' ';
        ply << "255 255 255" << endl;
    }

    ply.close();
}

// IO configs:
const string input_bal_filename = "../dataset/problem-16-22106-pre.txt";
const string output_ply_initial = "data/init.ply";
const string output_ply_optimized = "data/optimized.ply";

int main() {
    // load BAL dataset:
    BALDataset dataset(input_bal_filename);

    // save initial point clouds:
    dataset.write_to_ply(output_ply_initial);

    g2o::SparseOptimizer optimizer;
    build_optimizer(optimizer);
    build_optimization_graph(dataset, &optimizer);
    solve(optimizer);

    write_to_ply(
        &optimizer, 
        dataset.get_num_cameras(), dataset.get_num_points(),
        output_ply_optimized
    );
    
    return 0;
}