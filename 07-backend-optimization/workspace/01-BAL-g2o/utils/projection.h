#pragma once

#include "rotation.h"

namespace utils {
    namespace projection {
        // camera : 9 dims array with 
        // [0-2] : angle-axis rotation 
        // [3-5] : translateion
        // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
        // point : 3D location. 
        // predictions : 2D predictions with center of the image plane. 
        template<typename T>
        bool world_point_projection_with_distortion(const T *camera, const T *point_world, T *projections) {
            // point in camera frame:
            T point_camera[3];
            // p_camera = R*p_world + t
            utils::rotation::rotate_point_around_angle_axis(camera, point_world, point_camera);
            point_camera[0] += camera[3]; 
            point_camera[1] += camera[4]; 
            point_camera[2] += camera[5];

            // projection in normalized plane:
            T u = -point_camera[0]/point_camera[2];
            T v = -point_camera[1]/point_camera[2];

            // radial distortion factor:
            const T& k1 = camera[7];
            const T& k2 = camera[8];

            T r_squared = u*u + v*v;
            T distortion_factor = T(1.0) + r_squared*(k1 + k2*r_squared);

            // projection in pixel plane:
            const T& focal = camera[6];
            projections[0] = focal * distortion_factor * u;
            projections[1] = focal * distortion_factor * v;

            return true;
        }

        template<typename T>
        bool camera_point_projection_with_distortion(const T *point_camera, const T *camera_params, T* projections) {
            // projection in normalized plane:
            T u = -point_camera[0]/point_camera[2];
            T v = -point_camera[1]/point_camera[2];

            // radial distortion factor:
            const T& k1 = camera_params[1];
            const T& k2 = camera_params[2];

            T r_squared = u*u + v*v;
            T distortion_factor = T(1.0) + r_squared*(k1 + k2*r_squared);

            // projection in pixel plane:
            const T& focal = camera_params[0];
            projections[0] = focal * distortion_factor * u;
            projections[1] = focal * distortion_factor * v;

            return true;
        }
    }
}