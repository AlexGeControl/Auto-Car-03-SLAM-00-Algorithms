#include <Eigen/Core>
#include <Eigen/Geometry>

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/slam3d/se3quat.h"

#include "rotation.h"
#include "projection.h"

struct CameraBAL {
    // camera pose:
    g2o::SE3Quat T;
    // intrinsic:
    double f;
    // radial distortion:
    double k1, k2;
};

class VertexCameraBAL : public g2o::BaseVertex<9, CameraBAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCameraBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl (const double* update)
    {   
        // update pose:
        Eigen::Map<const g2o::Vector6d> dT(update);
        _estimate.T = g2o::SE3Quat::exp(dT)*_estimate.T;
        // update intrinsic:
        _estimate.f += update[6];
        // update radial distortion:
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];     
    }
};


class VertexPointBAL : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPointBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        Eigen::Vector3d::ConstMapType v(update);
        _estimate += v;
    }
};

class EdgeObservationBAL : public g2o::BaseBinaryEdge<2, Eigen::Vector2d,VertexCameraBAL, VertexPointBAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeObservationBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void computeError() override   // The virtual function comes from the Edge base class. Must define if you use edge.
    {
        const VertexCameraBAL* vertex_camera = static_cast<const VertexCameraBAL*> (vertex( 0 ));
        const VertexPointBAL* vertex_point = static_cast<const VertexPointBAL*> (vertex( 1 ));

        const Eigen::Vector3d point_camera = vertex_camera->estimate().T.map(vertex_point->estimate());
        const double camera_params[3] = {
            vertex_camera->estimate().f, 
            vertex_camera->estimate().k1,
            vertex_camera->estimate().k2
        };

        (*this)(point_camera.data(), camera_params, _error.data());
    }

    template<typename T>
    bool operator( ) (const T *point_camera, const T *camera_params, T* residuals) const
    {
        T projections[2];
        utils::projection::camera_point_projection_with_distortion(
            point_camera, camera_params, projections
        );

        residuals[0] = T(measurement()(0)) - projections[0];
        residuals[1] = T(measurement()(1)) - projections[1];

        return true;
    }

    virtual void linearizeOplus() override
    {   
        //
        // 1. use analytical Jacobian
        // this implementation mimics that of g2o::sba::VertexSE3ExpMap
        // the required Jacobians can be derived automatically using the accompany Jupyter notebook
        // using SymPy
        //
        const VertexCameraBAL *vertex_camera = static_cast<const VertexCameraBAL*>(vertex(0));
        const VertexPointBAL *vertex_point = static_cast<const VertexPointBAL*>(vertex(1));

        // camera pose:
        const g2o::SE3Quat &T = vertex_camera->estimate().T;
        // intrinsic:
        double f = vertex_camera->estimate().f;
        // radial distortion:
        double k1 = vertex_camera->estimate().k1;
        double k2 = vertex_camera->estimate().k2;
        // point in world frame:
        const Eigen::Vector3d &p_world = vertex_point->estimate();
        // point in camera frame:
        Eigen::Vector3d p_camera = T.map(p_world);
        double x = p_camera.x();
        double x_2 = x*x;
        double y = p_camera.y();
        double y_2 = y*y;
        double z = p_camera.z();
        double z_2 = z*z; double z_3 = z*z_2; double z_4 = z_2*z_2; double z_5 = z*z_4; double z_6 = z_3*z_3;
        double r_2 = (x_2 + y_2);
        double r_4 = r_2*r_2;

        // Jacobian for camera:
        _jacobianOplusXi(0,0) = -f*x*y*(z_4 + 2*z_2*(k1*z_2 + 2*k2*r_2) + r_2*(k1*z_2 + k2*r_2) + 2*r_2*(k1*z_2 + 2*k2*r_2))/z_6;
        _jacobianOplusXi(0,1) = f*(x_2*(z_4 + r_2*(k1*z_2 + k2*r_2) + 2*r_2*(k1*z_2 + 2*k2*r_2)) + z_2*(2*x_2*(k1*z_2 + 2*k2*r_2) + z_4 + r_2*(k1*z_2 + k2*r_2)))/z_6;
        _jacobianOplusXi(0,2) = -f*y*(z_4 + r_2*(k1*z_2 + k2*r_2))/z_5;
        _jacobianOplusXi(0,3) = f*(2*x_2*(k1*z_2 + 2*k2*r_2) + z_4 + r_2*(k1*z_2 + k2*r_2))/z_5;
        _jacobianOplusXi(0,4) = 2*f*x*y*(k1*z_2 + 2*k2*r_2)/z_5;
        _jacobianOplusXi(0,5) = -f*x*(z_4 + r_2*(k1*z_2 + k2*r_2) + 2*r_2*(k1*z_2 + 2*k2*r_2))/z_6;
        _jacobianOplusXi(0,6) = x*(z_4 + r_2*(k1*z_2 + k2*r_2))/z_5;
        _jacobianOplusXi(0,7) = f*x*r_2/z_3;
        _jacobianOplusXi(0,8) = f*x*r_4/z_5;

        _jacobianOplusXi(1,0) = -f*(y_2*(z_4 + r_2*(k1*z_2 + k2*r_2) + 2*r_2*(k1*z_2 + 2*k2*r_2)) + z_2*(2*y_2*(k1*z_2 + 2*k2*r_2) + z_4 + r_2*(k1*z_2 + k2*r_2)))/z_6;
        _jacobianOplusXi(1,1) = f*x*y*(z_4 + 2*z_2*(k1*z_2 + 2*k2*r_2) + r_2*(k1*z_2 + k2*r_2) + 2*r_2*(k1*z_2 + 2*k2*r_2))/z_6;
        _jacobianOplusXi(1,2) = f*x*(z_4 + r_2*(k1*z_2 + k2*r_2))/z_5;
        _jacobianOplusXi(1,3) = 2*f*x*y*(k1*z_2 + 2*k2*r_2)/z_5;
        _jacobianOplusXi(1,4) = f*(2*y_2*(k1*z_2 + 2*k2*r_2) + z_4 + r_2*(k1*z_2 + k2*r_2))/z_5;
        _jacobianOplusXi(1,5) = -f*y*(z_4 + r_2*(k1*z_2 + k2*r_2) + 2*r_2*(k1*z_2 + 2*k2*r_2))/z_6;
        _jacobianOplusXi(1,6) = y*(z_4 + r_2*(k1*z_2 + k2*r_2))/z_5;
        _jacobianOplusXi(1,7) = f*y*r_2/z_3;
        _jacobianOplusXi(1,8) = f*y*r_4/z_5;

        // Jacobian for point:
        Eigen::Matrix<double,2,3> P;
        P(0,0) = f*(2*x_2*(k1*z_2 + 2*k2*r_2) + z_4 + r_2*(k1*z_2 + k2*r_2))/z_5;
        P(0,1) = 2*f*x*y*(k1*z_2 + 2*k2*r_2)/z_5;
        P(0,2) = -f*x*(z_4 + r_2*(k1*z_2 + k2*r_2) + 2*r_2*(k1*z_2 + 2*k2*r_2))/z_6;

        P(1,0) = 2*f*x*y*(k1*z_2 + 2*k2*r_2)/z_5;
        P(1,1) = f*(2*y_2*(k1*z_2 + 2*k2*r_2) + z_4 + r_2*(k1*z_2 + k2*r_2))/z_5;
        P(1,2) = -f*y*(z_4 + r_2*(k1*z_2 + k2*r_2) + 2*r_2*(k1*z_2 + 2*k2*r_2))/z_6;
  
        _jacobianOplusXj = P * T.rotation().toRotationMatrix();
    }
};