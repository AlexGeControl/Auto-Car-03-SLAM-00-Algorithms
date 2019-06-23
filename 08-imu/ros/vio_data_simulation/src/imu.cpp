//
// Created by hyj on 18-1-19.
//

#include "imu.h"
#include "utilities.h"

// euler2Rotation:   body frame to interitail frame
Eigen::Matrix3d euler2Rotation( Eigen::Vector3d  eulerAngles)
{
    double roll = eulerAngles(0);
    double pitch = eulerAngles(1);
    double yaw = eulerAngles(2);

    double cr = cos(roll); double sr = sin(roll);
    double cp = cos(pitch); double sp = sin(pitch);
    double cy = cos(yaw); double sy = sin(yaw);

    Eigen::Matrix3d RIb;
    RIb<< cy*cp ,   cy*sp*sr - sy*cr,   sy*sr + cy* cr*sp,
            sy*cp,    cy *cr + sy*sr*sp,  sp*sy*cr - cy*sr,
            -sp,         cp*sr,           cp*cr;
    return RIb;
}

Eigen::Matrix3d eulerRates2bodyRates(Eigen::Vector3d eulerAngles)
{
    double roll = eulerAngles(0);
    double pitch = eulerAngles(1);

    double cr = cos(roll); double sr = sin(roll);
    double cp = cos(pitch); double sp = sin(pitch);

    Eigen::Matrix3d R;
    R<<  1,   0,    -sp,
            0,   cr,   sr*cp,
            0,   -sr,  cr*cp;

    return R;
}


IMU::IMU(Param p): param_(p)
{
    gyro_bias_ = Eigen::Vector3d::Zero();
    acc_bias_ = Eigen::Vector3d::Zero();
}

void IMU::addIMUnoise(MotionData& data)
{
    std::random_device rd;
    std::default_random_engine generator_(rd());
    std::normal_distribution<double> noise(0.0, 1.0);

    Eigen::Vector3d noise_gyro(noise(generator_),noise(generator_),noise(generator_));
    Eigen::Matrix3d gyro_sqrt_cov = param_.gyro_noise_sigma * Eigen::Matrix3d::Identity();
    data.imu_gyro = data.imu_gyro + gyro_sqrt_cov * noise_gyro / sqrt( param_.imu_timestep ) + gyro_bias_;

    Eigen::Vector3d noise_acc(noise(generator_),noise(generator_),noise(generator_));
    Eigen::Matrix3d acc_sqrt_cov = param_.acc_noise_sigma * Eigen::Matrix3d::Identity();
    data.imu_acc = data.imu_acc + acc_sqrt_cov * noise_acc / sqrt( param_.imu_timestep ) + acc_bias_;

    // gyro_bias update
    Eigen::Vector3d noise_gyro_bias(noise(generator_),noise(generator_),noise(generator_));
    gyro_bias_ += param_.gyro_bias_sigma * sqrt(param_.imu_timestep ) * noise_gyro_bias;
    data.imu_gyro_bias = gyro_bias_;

    // acc_bias update
    Eigen::Vector3d noise_acc_bias(noise(generator_),noise(generator_),noise(generator_));
    acc_bias_ += param_.acc_bias_sigma * sqrt(param_.imu_timestep ) * noise_acc_bias;
    data.imu_acc_bias = acc_bias_;

}

MotionData IMU::MotionModel(double t)
{

    MotionData data;
    // param
    float ellipse_x = 15;
    float ellipse_y = 20;
    float z = 1;           // z轴做sin运动
    float K1 = 10;          // z轴的正弦频率是x，y的k1倍
    float K = M_PI/ 10;    // 20 * K = 2pi 　　由于我们采取的是时间是20s, 系数K控制yaw正好旋转一圈，运动一周

    // translation
    // twb:  body frame in world frame
    Eigen::Vector3d position( ellipse_x * cos( K * t) + 5, ellipse_y * sin( K * t) + 5,  z * sin( K1 * K * t ) + 5);
    /*    
    Eigen::Vector3d dp(- K * ellipse_x * sin(K*t),  K * ellipse_y * cos(K*t), z*K1*K * cos(K1 * K * t));              // position导数　in world frame
    double K2 = K*K;
    Eigen::Vector3d ddp( -K2 * ellipse_x * cos(K*t),  -K2 * ellipse_y * sin(K*t), -z*K1*K1*K2 * sin(K1 * K * t));     // position二阶导数
    */
    Eigen::Vector3d dp(0,  0, 0);       // position导数　in world frame
    Eigen::Vector3d ddp( 0,  0, 0);     // position二阶导数

    // Rotation
    double k_roll = 0.1;
    double k_pitch = 0.2;
    //Eigen::Vector3d eulerAngles(k_roll * cos(t) , k_pitch * sin(t) , K*t );   // roll ~ [-0.2, 0.2], pitch ~ [-0.3, 0.3], yaw ~ [0,2pi]
    //Eigen::Vector3d eulerAnglesRates(-k_roll * sin(t) , k_pitch * cos(t) , K);      // euler angles 的导数
    Eigen::Vector3d eulerAngles(k_roll * sin(t) , k_pitch * sin(t) , K*t );   // roll ~ [-0.2, 0.2], pitch ~ [-0.3, 0.3], yaw ~ [0,2pi]
    Eigen::Vector3d eulerAnglesRates(0 , 0 , 0);      // euler angles 的导数

    Eigen::Matrix3d Rwb = euler2Rotation(eulerAngles);         // body frame to world frame
    Eigen::Vector3d imu_gyro = eulerRates2bodyRates(eulerAngles) * eulerAnglesRates;   //  euler rates trans to body gyro

    Eigen::Vector3d gn (0,0,-9.81);                                   //  gravity in navigation frame(ENU)   ENU (0,0,-9.81)  NED(0,0,9,81)
    Eigen::Vector3d imu_acc = Rwb.transpose() * ( ddp -  gn );  //  Rbw * Rwn * gn = gs

    data.imu_gyro = imu_gyro;
    data.imu_acc = imu_acc;
    data.Rwb = Rwb;
    data.twb = position;
    data.imu_velocity = dp;
    data.timestamp = t;

    return data;
}

void integrateEuler(
    // inputs: pose trajectory, step k
    std::vector<MotionData> &imudata, double dt, int k, const Eigen::Vector3d &gw, 
    // outputs: new position, velocity and quaternion
    Eigen::Vector3d &Pwb, Eigen::Vector3d &Vw, Eigen::Quaterniond &Qwb
) {
    // get IMU measurement at timestamp k:
    MotionData imupose = imudata[k - 1];

    // construct dq:
    Eigen::Quaterniond dq;
    Eigen::Vector3d dtheta_half = 0.5 * imupose.imu_gyro * dt;
    dq.w() = 1;
    dq.x() = dtheta_half.x();
    dq.y() = dtheta_half.y();
    dq.z() = dtheta_half.z();

    // construct a:
    Eigen::Vector3d acc_w = Qwb * (imupose.imu_acc) + gw;

    // update q:
    Qwb = Qwb * dq;
    // update v:
    Vw = Vw + acc_w * dt;
    // update p:
    Pwb = Pwb + Vw * dt + 0.5 * dt * dt * acc_w;
}

void integrateMidPoint(
    // inputs: pose trajectory, step k
    std::vector<MotionData> &imudata, double dt, int k, const Eigen::Vector3d &gw, 
    // outputs: new position, velocity and quaternion
    Eigen::Vector3d &Pwb, Eigen::Vector3d &Vw, Eigen::Quaterniond &Qwb
) {
    // get IMU measurement at timestamp k & k + 1:
    MotionData imupose_curr = imudata[k - 1];
    MotionData imupose_next = imudata[k + 0];

    // construct dq:
    Eigen::Quaterniond dq;
    Eigen::Vector3d dtheta_half = 0.5 * (0.5 * (imupose_curr.imu_gyro + imupose_next.imu_gyro))* dt;
    dq.w() = 1;
    dq.x() = dtheta_half.x();
    dq.y() = dtheta_half.y();
    dq.z() = dtheta_half.z();

    // construct a:
    Eigen::Quaterniond Qwb_curr = Qwb;
    Eigen::Quaterniond Qwb_next = Qwb * dq;
    Eigen::Vector3d acc_w = 0.5 * (
        // current:
        (Qwb_curr * (imupose_curr.imu_acc) + gw)
        + 
        // next:
        (Qwb_next * (imupose_next.imu_acc) + gw)
    );

    // update q:
    Qwb = Qwb_next;
    // update v:
    Vw = Vw + acc_w * dt;
    // update p:
    Pwb = Pwb + Vw * dt + 0.5 * dt * dt * acc_w;   
}

//读取生成的imu数据并用imu动力学模型对数据进行计算，最后保存imu积分以后的轨迹，
//用来验证数据以及模型的有效性。
void IMU::testImu(std::string src, std::string dist)
{
    std::vector<MotionData>imudata;
    LoadPose(src,imudata);

    std::ofstream save_points;
    save_points.open(dist);

    double dt = param_.imu_timestep;
    Eigen::Vector3d Pwb = init_twb_;              // position :    from  imu measurements
    Eigen::Quaterniond Qwb(init_Rwb_);            // quaterniond:  from imu measurements
    Eigen::Vector3d Vw = init_velocity_;          // velocity  :   from imu measurements
    Eigen::Vector3d gw(0,0,-9.81);                // ENU frame
    for (int i = 1; i < imudata.size(); ++i) {
        // imu 动力学模型:
        #ifdef MOTION_UPDATE_EULER
            // a. 欧拉积分
            integrateEuler(imudata, dt, i, gw, Pwb, Vw, Qwb);
        #else
            // b. 中值积分
            integrateMidPoint(imudata, dt, i, gw, Pwb, Vw, Qwb);
        #endif

        //　按着imu postion, imu quaternion , cam postion, cam quaternion 的格式存储，由于没有cam，所以imu存了两次
        save_points << imudata[i].timestamp << " "
                    << Qwb.w() << " "
                    << Qwb.x() << " "
                    << Qwb.y() << " "
                    << Qwb.z() << " "
                    << Pwb(0) << " "
                    << Pwb(1) << " "
                    << Pwb(2) << " "
                    << Qwb.w() << " "
                    << Qwb.x() << " "
                    << Qwb.y() << " "
                    << Qwb.z() << " "
                    << Pwb(0) << " "
                    << Pwb(1) << " "
                    << Pwb(2) << " "
                    << std::endl;
    }

    std::cout<<"test　end"<<std::endl;
}