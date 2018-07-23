//
// Created by xiang on 1/4/18.
// this program shows how to perform direct bundle adjustment
//
#include <iostream>

#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <Eigen/Core>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <pangolin/pangolin.h>
#include <boost/format.hpp>

using namespace std;

typedef vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> VecSE3;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;

// global variables
string pose_file = "./poses.txt";
string points_file = "./points.txt";

// intrinsics
double fx = 277.34;
double fy = 291.402;
double cx = 312.234;
double cy = 239.777;

class Image {
public:
    // constructor:
    Image(const cv::Mat &image) {
        // intensity:
        intensity = image.clone();
        // gradient:
        Scharr(image, grad_x, CV_64F, 1, 0);
        Scharr(image, grad_y, CV_64F, 0, 1);         
    }

    int cols() const {
        return intensity.cols;
    }

    int rows() const {
        return intensity.rows;
    }

    // bilinear interpolation -- pixel intensity:
    double get_pixel_value(double x, double y) const {
        double i11 = intensity.at<uchar>(int(y) + 0, int(x) + 0);
        double i12 = intensity.at<uchar>(int(y) + 1, int(x) + 0);
        double i21 = intensity.at<uchar>(int(y) + 0, int(x) + 1);
        double i22 = intensity.at<uchar>(int(y) + 1, int(x) + 1);

        double xx = x - floor(x);
        double yy = y - floor(y);

        return double(
            (1.0 - xx)*(1.0 - yy)*i11 +
            (0.0 + xx)*(1.0 - yy)*i21 + 
            (1.0 - xx)*(0.0 + yy)*i12 + 
            (0.0 + xx)*(0.0 + yy)*i22
        );
    }

    Eigen::Vector2d get_gradient(double x, double y) const {
        // extract gradient:
        Eigen::Vector2d gradient(
            get_grad_value(grad_x, x, y),
            get_grad_value(grad_y, x, y)
        );
        gradient /= 26.0;

        return gradient;
    }

private:
    // intensity:
    cv::Mat intensity;
    // gradient:
    cv::Mat grad_x, grad_y;

    // bilinear interpolation -- gradient value: 
    double get_grad_value(const cv::Mat &grad, double x, double y) const {
        double g11 = grad.at<double>(int(y) + 0, int(x) + 0);
        double g12 = grad.at<double>(int(y) + 1, int(x) + 0);
        double g21 = grad.at<double>(int(y) + 0, int(x) + 1);
        double g22 = grad.at<double>(int(y) + 1, int(x) + 1);

        double xx = x - floor(x);
        double yy = y - floor(y);

        return double(
            (1.0 - xx)*(1.0 - yy)*g11 +
            (0.0 + xx)*(1.0 - yy)*g21 + 
            (1.0 - xx)*(0.0 + yy)*g12 + 
            (0.0 + xx)*(0.0 + yy)*g22
        );
    }
};

inline void camera_point_projection(
    const Sophus::SE3 &se3, 
    const Eigen::Vector3d &point,
    double *projections
) {
    // projection in camera frame:
    const Eigen::Vector3d point_camera = se3*point;
    // projection in normalized plane:
    double x_prime = point_camera.x() / point_camera.z();
    double y_prime = point_camera.y() / point_camera.z();
    // projection in pixel plane:
    projections[0] = fx*x_prime + cx;
    projections[1] = fy*y_prime + cy;
}

// g2o vertex that use sophus::SE3 as pose
class VertexSophus : public g2o::BaseVertex<6, Sophus::SE3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSophus() {}

    ~VertexSophus() {}

    bool read(std::istream &is) {}

    bool write(std::ostream &os) const {}

    virtual void setToOriginImpl() {
        _estimate = Sophus::SE3();
    }

    virtual void oplusImpl(const double *update_) {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
        setEstimate(Sophus::SE3::exp(update) * estimate());
    }
};

// TODO edge of projection error, implement it
// 16x1 error, which is the errors in patch
typedef Eigen::Matrix<double,16,1> Vector16d;
class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vector16d, VertexSophus, g2o::VertexSBAPointXYZ> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    static const int HALF_PATCH_SIZE = 2;
    static const int FULL_PATCH_SIZE = 4;
    const int DELTA[FULL_PATCH_SIZE] = {-2, -1, +0, +1};

    EdgeDirectProjection(const Image &image): image(image)  {
    }

    ~EdgeDirectProjection() {}

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

    virtual void computeError() override {
        const VertexSophus* vertex_camera = static_cast<const VertexSophus*>(vertex(0));
        const g2o::VertexSBAPointXYZ* vertex_point = static_cast<const g2o::VertexSBAPointXYZ*>(vertex(1));  

        // project to pixel frame:
        camera_point_projection(
            vertex_camera->estimate(), vertex_point->estimate(),
            projections
        );

        // calculate error:
        double u = projections[0]; double v = projections[1];
        for (int i = 0; i < FULL_PATCH_SIZE; ++i) {
            for (int j = 0; j < FULL_PATCH_SIZE; ++j) {
                const int idx = FULL_PATCH_SIZE*i + j;
                _error(idx) = measurement()(idx) - image.get_pixel_value(
                    u + DELTA[i], v + DELTA[j]
                );
            }
        } 
    }

    virtual void linearizeOplus() override
    {   
        //
        // 1. use analytical Jacobian
        // this implementation mimics that of g2o::sba::VertexSE3ExpMap
        // the required Jacobians can be derived automatically using the accompany Jupyter notebook
        // using SymPy
        //
        const VertexSophus* vertex_camera = static_cast<const VertexSophus*>(vertex(0));
        const g2o::VertexSBAPointXYZ* vertex_point = static_cast<const g2o::VertexSBAPointXYZ*>(vertex(1));  

        // project to camera frame:
        const Eigen::Vector3d point_camera = (vertex_camera->estimate())*(vertex_point->estimate());
        double x = point_camera.x();
        double x_2 = x*x;
        double y = point_camera.y();
        double y_2 = y*y;
        double z = point_camera.z();
        double z_2 = z*z;

        // project to pixel frame:
        camera_point_projection(
            vertex_camera->estimate(), vertex_point->estimate(),
            projections
        );

        // Jacobian of se3:
        Eigen::Matrix<double, 2, 6> J_se3;
        J_se3(0,0) = fx*x*y/z_2;
        J_se3(0,1) = -fx*x_2/z_2 - fx;
        J_se3(0,2) = fx*y/z;
        J_se3(0,3) = -fx/z;
        J_se3(0,4) = 0;
        J_se3(0,5) = fx*x/z_2;

        J_se3(1,0) = fy*y_2/z_2 + fy;
        J_se3(1,1) = -fy*x*y/z_2;
        J_se3(1,2) = -fy*x/z;
        J_se3(1,3) = 0;
        J_se3(1,4) = -fy/z;
        J_se3(1,5) = fy*y/z_2;

        // Jacobian of point:
        Eigen::Matrix<double, 2, 3> J_P;
        J_P(0,0) = -fx/z;
        J_P(0,1) = 0;
        J_P(0,2) = fx*x/z_2;

        J_P(1,0) = 0;
        J_P(1,1) = -fy/z;
        J_P(1,2) = fy*y/z_2;
        J_P = J_P*(vertex_camera->estimate().rotation_matrix());

        // image gradient:
        Eigen::Matrix<double, 16, 2> J_i;
        double u = projections[0]; double v = projections[1];
        for (int i = 0; i < FULL_PATCH_SIZE; ++i) {
            for (int j = 0; j < FULL_PATCH_SIZE; ++j) {
                const int idx = FULL_PATCH_SIZE*i + j;
                const Eigen::Vector2d J_i_ = image.get_gradient(
                    u + DELTA[i], v + DELTA[j]
                );
                J_i(idx, 0) = J_i_(0);
                J_i(idx, 1) = J_i_(1);
            }
        }

        _jacobianOplusXi = J_i*J_se3;
        _jacobianOplusXj = J_i*J_P;   
    }
private:
    const Image &image;
    double projections[2];
};

/**
    build optimization graph for BAL.

    @param camera the camera instance in BAL format.
    @param result the output camera center in world frame.
*/ 
void build_optimization_graph(
    g2o::SparseOptimizer* optimizer, 
    const VecSE3 &poses, const vector<Image> &images,
    const VecVec3d &points, const vector<double *> &color
);

void get_optimization_results(
    g2o::SparseOptimizer* optimizer, 
    VecSE3 &poses,
    VecVec3d &points
);

// plot the poses and points for you, need pangolin
void Draw(const VecSE3 &poses, const VecVec3d &points);

int main(int argc, char **argv) {
    // read poses:
    VecSE3 poses;
    ifstream fin(pose_file);
    while (!fin.eof()) {
        double timestamp = 0;
        fin >> timestamp;
        if (timestamp == 0) break;
        double data[7];
        for (auto &d: data) fin >> d;
        poses.push_back(Sophus::SE3(
                Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                Eigen::Vector3d(data[0], data[1], data[2])
        ));
        if (!fin.good()) break;
    }
    fin.close();
    // read images
    vector<Image> images;
    boost::format fmt("./%d.png");
    for (int i = 0; i < 7; i++) {
        images.push_back(
            Image(cv::imread((fmt % i).str(), 0))
        );
    }

    // read points:
    VecVec3d points;
    vector<double *> color;
    fin.open(points_file);
    while (!fin.eof()) {
        double xyz[3] = {0};
        for (int i = 0; i < 3; i++) fin >> xyz[i];
        if (xyz[0] == 0) break;
        points.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]));
        double *c = new double[16];
        for (int i = 0; i < 16; i++) fin >> c[i];
        color.push_back(c);

        if (fin.good() == false) break;
    }
    fin.close();

    cout << "poses: " << poses.size() << ", points: " << points.size() << endl;

    // build optimization problem
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> DirectBlock;  // 求解的向量是6＊1的
    DirectBlock::LinearSolverType *linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
    DirectBlock *solver_ptr = new DirectBlock(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // add vertices, edges into the graph optimizer
    build_optimization_graph(
        &optimizer, 
        poses, images,
        points, color
    );
    // perform optimization
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(200);

    // fetch data from the optimizer
    get_optimization_results(&optimizer, poses, points);

    // plot the optimized points and poses
    Draw(poses, points);

    // delete color data
    for (auto &c: color) delete[] c;
    return 0;
}

void build_optimization_graph(
    g2o::SparseOptimizer* optimizer, 
    const VecSE3 &poses, const vector<Image> &images,
    const VecVec3d &points, const vector<double *> &color
) {
    // consts:
    const int NUM_CAMERAS = poses.size();
    const int NUM_POINTS = points.size();
    const int POINT_ID_BASE = NUM_CAMERAS;

    // add camera vertices:
    for(int i = 0; i < poses.size(); ++i)
    {
        VertexSophus *vertex_camera = new VertexSophus();

        vertex_camera->setEstimate(poses[i]);
        vertex_camera->setId(i);

        optimizer->addVertex(vertex_camera);
    }

    // add point vertices:
    for(int i = 0; i < points.size(); ++i)
    {
        g2o::VertexSBAPointXYZ *vertex_point = new g2o::VertexSBAPointXYZ();

        vertex_point->setEstimate(points[i]);
        vertex_point->setId(POINT_ID_BASE + i);
        vertex_point->setMarginalized(true);

        optimizer->addVertex(vertex_point);
    }

    // add observation edges:
    double projections[2];
    double u, v;
    for(int i = 0; i < NUM_CAMERAS; ++i) {
        for(int j = 0; j < NUM_POINTS; ++j) {
            EdgeDirectProjection *edge_observation = new EdgeDirectProjection(images[i]);

            // get id for camera and point:
            const int camera_id = i; 
            const int point_id = POINT_ID_BASE + j; 

            VertexSophus* vertex_camera = dynamic_cast<VertexSophus*>(optimizer->vertex(camera_id));
            g2o::VertexSBAPointXYZ* vertex_point = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer->vertex(point_id));  

            // project to pixel frame:
            camera_point_projection(
                vertex_camera->estimate(), vertex_point->estimate(),
                projections
            );

            // validity check:
            u = projections[0]; v = projections[1];
            if (int(u) <= EdgeDirectProjection::HALF_PATCH_SIZE || int(u) >= images[i].cols() - EdgeDirectProjection::HALF_PATCH_SIZE ||
                int(v) <= EdgeDirectProjection::HALF_PATCH_SIZE || int(v) >= images[i].rows() - EdgeDirectProjection::HALF_PATCH_SIZE) {
                continue;
            } 

            // set the vertex by the ids for an edge observation
            edge_observation->setVertex(0, vertex_camera);
            edge_observation->setVertex(1, vertex_point);
            // information matrix:
            edge_observation->setInformation(
                Eigen::Matrix<double, 16, 16>::Identity()
            );
            // loss function:
            g2o::RobustKernelHuber *robust_kernel = new g2o::RobustKernelHuber;
            robust_kernel->setDelta(1.0);
            edge_observation->setRobustKernel(robust_kernel);
            // measurement:
            edge_observation->setMeasurement(
                Vector16d(color[j])
            );

            optimizer->addEdge(edge_observation) ;
        }
    }
}

void get_optimization_results(
    g2o::SparseOptimizer* optimizer, 
    VecSE3 &poses,
    VecVec3d &points
) {
    // consts:
    const int NUM_CAMERAS = poses.size();
    const int NUM_POINTS = points.size();
    const int POINT_ID_BASE = NUM_CAMERAS;

    // add camera vertices:
    for(int i = 0; i < NUM_CAMERAS; ++i)
    {
        const VertexSophus* vertex_camera = dynamic_cast<const VertexSophus*>(optimizer->vertex(i));
        const Sophus::SE3 &pose = vertex_camera->estimate();

        poses[i] = pose;
    }

    // add point vertices:
    for(int i = 0; i < NUM_POINTS; ++i)
    {
        const g2o::VertexSBAPointXYZ* vertex_point = dynamic_cast<const g2o::VertexSBAPointXYZ*>(optimizer->vertex(POINT_ID_BASE + i));
        const Eigen::Vector3d &point = vertex_point->estimate();

        points[i] = point;
    }
}

void Draw(const VecSE3 &poses, const VecVec3d &points) {
    if (poses.empty() || points.empty()) {
        cerr << "parameter is empty!" << endl;
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
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.1;
        int width = 640, height = 480;
        for (auto &Tcw: poses) {
            glPushMatrix();
            Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
            glMultMatrixf((GLfloat *) m.data());
            glColor3f(1, 0, 0);
            glLineWidth(2);
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glEnd();
            glPopMatrix();
        }

        // points
        glPointSize(2);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < points.size(); i++) {
            glColor3f(0.0, points[i][2]/4, 1.0-points[i][2]/4);
            glVertex3d(points[i][0], points[i][1], points[i][2]);
        }
        glEnd();

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

