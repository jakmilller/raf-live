#pragma once

// System includes for socket types
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/wrench.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// Custom interfaces
#include "raf_interfaces/srv/set_joint_angles.hpp"
#include "raf_interfaces/srv/set_joint_velocity.hpp"
#include "raf_interfaces/srv/set_joint_waypoints.hpp"
#include "raf_interfaces/srv/set_pose.hpp"
#include "raf_interfaces/srv/set_gripper.hpp"
#include "raf_interfaces/srv/set_twist.hpp"

// Kortex API
#include <KDetailedException.h>
#include <BaseClientRpc.h>
#include <BaseCyclicClientRpc.h>
#include <ActuatorConfigClientRpc.h>
#include <SessionClientRpc.h>
#include <SessionManager.h>
#include <RouterClient.h>
#include <TransportClientTcp.h>
#include <TransportClientUdp.h>

#include <chrono>
#include <functional>
#include <thread>
#include <atomic>

#define TCP_PORT 10000
#define UDP_PORT 10001

namespace k_api = Kinova::Api;

// Custom message type for cartesian state (can be simplified to use geometry_msgs if needed)
struct CartesianState {
    geometry_msgs::msg::Pose pose;
    geometry_msgs::msg::Twist twist;
    geometry_msgs::msg::Wrench wrench;
};

class Controller : public rclcpp::Node
{
public:
    Controller();
    ~Controller();

private:
    // ROS2 publishers
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr mJointStatePub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mCartesianStatePub;
    
    // ROS2 subscribers
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr mTareFTSensorSub;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr mEStopSub;
    
    // ROS2 services
    rclcpp::Service<raf_interfaces::srv::SetJointAngles>::SharedPtr mSetJointPositionService;
    rclcpp::Service<raf_interfaces::srv::SetJointVelocity>::SharedPtr mSetJointVelocityService;
    rclcpp::Service<raf_interfaces::srv::SetJointWaypoints>::SharedPtr mSetJointWaypointsService;
    rclcpp::Service<raf_interfaces::srv::SetPose>::SharedPtr mSetPoseService;
    rclcpp::Service<raf_interfaces::srv::SetGripper>::SharedPtr mSetGripperService;
    rclcpp::Service<raf_interfaces::srv::SetTwist>::SharedPtr mSetTwistService;
    
    // ROS2 timer
    rclcpp::TimerBase::SharedPtr mRobotStateTimer;
    
    // Service callbacks
    void setJointPosition(const std::shared_ptr<raf_interfaces::srv::SetJointAngles::Request> request,
                         std::shared_ptr<raf_interfaces::srv::SetJointAngles::Response> response);
    void setJointVelocity(const std::shared_ptr<raf_interfaces::srv::SetJointVelocity::Request> request,
                         std::shared_ptr<raf_interfaces::srv::SetJointVelocity::Response> response);
    void setJointWaypoints(const std::shared_ptr<raf_interfaces::srv::SetJointWaypoints::Request> request,
                          std::shared_ptr<raf_interfaces::srv::SetJointWaypoints::Response> response);
    void setPose(const std::shared_ptr<raf_interfaces::srv::SetPose::Request> request,
                std::shared_ptr<raf_interfaces::srv::SetPose::Response> response);
    void setGripper(const std::shared_ptr<raf_interfaces::srv::SetGripper::Request> request,
                   std::shared_ptr<raf_interfaces::srv::SetGripper::Response> response);
    void setTwist(const std::shared_ptr<raf_interfaces::srv::SetTwist::Request> request,
                 std::shared_ptr<raf_interfaces::srv::SetTwist::Response> response);
    
    // Subscriber callbacks
    void tareFTSensorCallback(const std_msgs::msg::Bool::SharedPtr msg);
    void eStopCallback(const std_msgs::msg::Bool::SharedPtr msg);
    
    // Timer callback
    void publishState();
    
    // Utility functions
    inline double degreesToRadians(double degrees);
    inline double radiansToDegrees(double radians);
    
    // Kortex API members (same as original)
    k_api::TransportClientTcp* m_tcp_transport;
    k_api::RouterClient* m_tcp_router;
    k_api::SessionManager* m_tcp_session_manager;
    
    k_api::Base::BaseClient* mBase;
    k_api::BaseCyclic::BaseCyclicClient* mBaseCyclic;
    k_api::ActuatorConfig::ActuatorConfigClient* mActuatorConfig;
    
    k_api::Base::ServoingModeInformation mServoingMode;
    k_api::ActuatorConfig::ControlModeInformation mControlModeMessage;
    k_api::BaseCyclic::Feedback mLastFeedback;
    
    // Connection parameters
    std::string m_username;
    std::string m_password;
    std::string m_ip_address;
    int m_api_session_inactivity_timeout_ms;
    int m_api_connection_inactivity_timeout_ms;
    
    // Force/torque sensor data
    std::atomic<bool> mTareFTSensor{false};
    std::atomic<bool> mUpdateForceThreshold{false};
    std::vector<double> mZeroFTSensorValues;
    std::vector<double> mFTSensorValues;
    std::vector<double> mForceThreshold;
    std::vector<double> mNewForceThreshold;
    
    bool mWatchdogActive;
    
    // Timeout for actions
    static constexpr auto TIMEOUT_DURATION = std::chrono::seconds(20);
};