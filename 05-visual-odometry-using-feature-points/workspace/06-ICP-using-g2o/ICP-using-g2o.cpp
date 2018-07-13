#include <unistd.h>
#include <chrono>
// c++ standard library:
#include <string>
#include <iostream>
#include <fstream>
// eigen for matrix algebra:
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
// matrix lie algebra:
#include <sophus/se3.h>
// graph optimization:
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
// pangolin for visualization
#include <pangolin/pangolin.h>

using namespace std;

// path to aligned trajectories file:
string aligned_trajectories_file = "./compare.txt";

void LoadTrajectory(
    const string &aligned_trajectories_file,
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> &estimated,
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> &ground_truth     
);

class EdgeProjectXYZ2XYZ: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectXYZ2XYZ(const Eigen::Vector3d &point): _point(point) {}

    virtual void computeError() {
        const g2o::VertexSE3Expmap *pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);

        _error = _measurement - pose->estimate().map(_point);        
    }

    virtual void linearizeOplus() {
        const g2o::VertexSE3Expmap *pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);

        Eigen::Vector3d point_estimation = pose->estimate().map(_point);

        double x = point_estimation.x();
        double y = point_estimation.y(); 
        double z = point_estimation.z(); 

        _jacobianOplusXi(0, 0) = 0.0; 
        _jacobianOplusXi(0, 1) = -z;          
        _jacobianOplusXi(0, 2) = +y; 
        _jacobianOplusXi(0, 3) = -1;   
        _jacobianOplusXi(0, 4) = 0.0; 
        _jacobianOplusXi(0, 5) = 0.0;   

        _jacobianOplusXi(1, 0) = +z; 
        _jacobianOplusXi(1, 1) = 0.0;          
        _jacobianOplusXi(1, 2) = -x; 
        _jacobianOplusXi(1, 3) = 0.0;   
        _jacobianOplusXi(1, 4) = -1; 
        _jacobianOplusXi(1, 5) = 0.0; 

        _jacobianOplusXi(2, 0) = -y; 
        _jacobianOplusXi(2, 1) = +x;          
        _jacobianOplusXi(2, 2) = 0.0; 
        _jacobianOplusXi(2, 3) = 0.0;   
        _jacobianOplusXi(2, 4) = 0.0; 
        _jacobianOplusXi(2, 5) = -1; 
    }

    bool read(istream &in) {}
    bool write(ostream &out) const {}
private:
    Eigen::Vector3d _point;
};

void EstimateTransform(
    const vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> &estimated,
    const vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> &ground_truth,
    Eigen::Matrix3d &R, Eigen::Vector3d &t    
);

// function for plotting trajectory, don't edit this code
// start point is red and end point is blue
void DrawTrajectory(
    const vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> &estimated,
    const vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> &ground_truth,
    const Eigen::Matrix3d &R, const Eigen::Vector3d &t    
);

int main(int argc, char **argv) {
    // estimated and ground truth trajectories:
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> estimated;
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> ground_truth;

    // load trajectories:
    LoadTrajectory(
        aligned_trajectories_file,
        estimated, ground_truth     
    );

    // estimate transform:
    Eigen::Matrix3d R; Eigen::Vector3d t;
    EstimateTransform(estimated, ground_truth, R, t);

    // draw trajectory in pangolin
    DrawTrajectory(estimated, ground_truth, R, t);  
    
    return 0;
}

/*******************************************************************************************/
void LoadTrajectory(
    const string &aligned_trajectories_file,
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> &estimated,
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> &ground_truth     
) {
    // memory buffer:
    Eigen::Vector3d te, tg;
    Eigen::Quaterniond qe, qg;

    // trajectory reader:
    ifstream trajectory(aligned_trajectories_file);
    const int D = 16;
    double value[D]; int d = 0;

    while (trajectory >> value[d]) {
        // update index:
        d = (d + 1) % D;

        if (0 == d) {
            // estimated pose:
            te.x() = value[1];
            te.y() = value[2];
            te.z() = value[3];
            qe.x() = value[4];
            qe.y() = value[5];
            qe.z() = value[6];
            qe.w() = value[7];
            // ground truth pose:
            tg.x() = value[9];
            tg.y() = value[10];
            tg.z() = value[11];
            qg.x() = value[12];
            qg.y() = value[13];
            qg.z() = value[14];
            qg.w() = value[15];           

            estimated.push_back(Sophus::SE3(qe, te));
            ground_truth.push_back(Sophus::SE3(qg, tg));
        }
    }
};

void EstimateTransform(
    const vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> &estimated,
    const vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> &ground_truth,
    Eigen::Matrix3d &R, Eigen::Vector3d &t    
) {
    // total number of observations:
    const int N = estimated.size();
    
    // solver:
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolver;
    BlockSolver::LinearSolverType *linear_solver = new g2o::LinearSolverDense<BlockSolver::PoseMatrixType>();
    BlockSolver *block_solver = new BlockSolver(linear_solver);
    g2o::OptimizationAlgorithmLevenberg *algorithm = new g2o::OptimizationAlgorithmLevenberg(block_solver);
    // optimizer:
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    // set vertices:
    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(
        g2o::SE3Quat(
            R,
            t
        )        
    );
    optimizer.addVertex(pose);

    // set edges:
    for (int i = 0; i < N; ++i) {
        EdgeProjectXYZ2XYZ *point = new EdgeProjectXYZ2XYZ(estimated[i].translation());

        point->setId(i);
        point->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap*>(pose));
        point->setMeasurement(ground_truth[i].translation());
        point->setInformation(Eigen::Matrix3d::Identity());

        optimizer.addEdge(point);
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
    auto T = Eigen::Isometry3d(pose->estimate()).matrix();
    R << \
        T(0, 0), T(0, 1), T(0, 2),
        T(1, 0), T(1, 1), T(1, 2),
        T(2, 0), T(2, 1), T(2, 2);
    
    t << T(0, 3), T(1, 3), T(2, 3);
    cout << "[Optimized Pose]: " << endl << R << endl << t << endl;
}

void DrawTrajectory(
    const vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> &estimated,
    const vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> &ground_truth, 
    const Eigen::Matrix3d &R, const Eigen::Vector3d &t
) {
    if (estimated.empty() || ground_truth.empty() || estimated.size() != ground_truth.size()) {
        cerr << "The trajectories are not aligned!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        const int N = ground_truth.size();
        for (size_t i = 0; i < N - 1; i++) {
            // estimated:
            glColor3f(0.0f, 0.0f, 1.0f);
            glBegin(GL_LINES);
            auto te1 = R * estimated[i].translation() + t;
            auto te2 = R * estimated[i + 1].translation() + t;
            glVertex3d(te1[0], te1[1], te1[2]);
            glVertex3d(te2[0], te2[1], te2[2]);
            glEnd();
            // ground truth:
            glColor3f(1.0f, 0.0f, 0.0f);
            glBegin(GL_LINES);
            auto tg1 = ground_truth[i].translation();
            auto tg2 = ground_truth[i + 1].translation();
            glVertex3d(tg1[0], tg1[1], tg1[2]);
            glVertex3d(tg2[0], tg2[1], tg2[2]);
            glEnd();            
        }
        pangolin::FinishFrame();
    }
}