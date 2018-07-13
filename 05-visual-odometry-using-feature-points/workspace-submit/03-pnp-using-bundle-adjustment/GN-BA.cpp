
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <sophus/se3.h>

using namespace std;
using namespace Eigen;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

string p3d_file = "./p3d.txt";
string p2d_file = "./p2d.txt";

int main(int argc, char **argv) {
    // perspectives and points:
    VecVector3d p3d;
    VecVector2d p2d;
    // intrinsic params:
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // memory buffer for p3d and p2d
    double v[3];
    int D, d;

    // points:
    ifstream p3d_filestream(p3d_file);
    D = 3; d = 0;
    while (p3d_filestream >> v[d]) {
        // update index:
        d = (d + 1) % D;

        // update point:
        if (0 == d) {
            p3d.push_back(Vector3d(v[0], v[1], v[2]));
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
            p2d.push_back(Vector2d(v[0], v[1]));
        }
    }   
    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();
    cout << "points: " << nPoints << endl;

    Sophus::SE3 T_esti; // estimated pose

    for (int iter = 0; iter < iterations; iter++) {
        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < nPoints; i++) {
            // rigid transform defined by SE3:
            auto R = T_esti.rotation_matrix();
            auto t = T_esti.translation();
            // point in camera frame:
            auto p_camera = R * p3d[i] + t;

            // point in pixel frame:
            auto p_pixel = K * p_camera;

            // error:
            auto e = p2d[i];
            e.x() -= p_pixel.x() / p_pixel.z();
            e.y() -= p_pixel.y() / p_pixel.z();

	        // compute jacobian
            double X_prime = p_camera.x();
            double Y_prime = p_camera.y();
            double Z_prime = p_camera.z();
            Matrix<double, 2, 6> J;
            J << \
                                                  fx/Z_prime,                                           0.0, \
                               -fx*X_prime/(Z_prime*Z_prime),         -fx*X_prime*Y_prime/(Z_prime*Z_prime), \
                fx*(1 + (X_prime*X_prime)/(Z_prime*Z_prime)),                           -fx*Y_prime/Z_prime, \
                                                         0.0,                                    fy/Z_prime, \
                               -fy*Y_prime/(Z_prime*Z_prime), -fy*(1 + (Y_prime*Y_prime)/(Z_prime*Z_prime)), \
                      fy*(X_prime*Y_prime)/(Z_prime*Z_prime),                            fy*X_prime/Z_prime
            ;
            J *= -1.0;

            // update Hessian:
            H += J.transpose() * J;
            // update b:
            b += -J.transpose() * e;

            // update cost:
            cost += 0.5 * e.squaredNorm();
        }

	    // solve dx 
        Vector6d dx = H.ldlt().solve(b);

        // solution is nan: abort
        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        // increased cost: abort
        if (iter > 0 && cost >= lastCost) {
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        T_esti = Sophus::SE3::exp(dx) * T_esti;

        // display progress:
        lastCost = cost;
        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    
    return 0;
}
