#!/usr/bin/python3
import rospy
from landmark_localizer.localizer import BlockBotLocalizer
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from sensor_msgs.msg import JointState
from apriltag_ros.msg import AprilTagDetectionArray
from tf.transformations import euler_from_quaternion


class LocalizerListener:
    def __init__(self) -> None:
        self.robot_name = rospy.get_param('robot_name', 'locobot')
        self.reset_issued = True
        self.camera_tilt = 0
        self.landmark_tag_ids = rospy.get_param('landmark_tag_ids', [])
        rospy.init_node('localizer-listener')
    
        rospy.Subscriber(
            "/" + self.robot_name + "/mobile_base/commands/reset_odometry",
            Empty,
            self.odom_reset
        )

        rospy.Subscriber(
            "/" + self.robot_name + "/mobile_base/odom",
            Odometry,
            self.update_odom
        )

        rospy.Subscriber(
            "/tag_detections",
            AprilTagDetectionArray, 
            self.update_landmark_detections
        )

        rospy.Subscriber(
            "/" + self.robot_name + "/joint_states",
            JointState, 
            self.update_camera_tilt
        )

        self.estimate_pub = rospy.Publisher(
            'estimated_pose',
            Pose2D,
            queue_size=1
        )

    def odom_reset(self, _):
        '''Odometry has been reset'''
        self.reset_issued = True

    def reset_localizer(self):
        print("Resetting Localizer")
        self.localizer = BlockBotLocalizer()
        self.odom = None
        self.landmark_detections = []

    def update_odom(self, msg):
        # Convert from quaternion (adapted from Interbotix code)
        odom = msg.pose.pose
        quat = (
            odom.orientation.x,
            odom.orientation.y,
            odom.orientation.z,
            odom.orientation.w
        )
        self.odom = [odom.position.x, odom.position.y, euler_from_quaternion(quat)[2]]

    def update_landmark_detections(self, msg: AprilTagDetectionArray):
        self.landmark_detections = [
            tag for tag in msg.detections if tag.id[0] in self.landmark_tag_ids
        ]

    def update_camera_tilt(self, msg):
        self.camera_tilt = msg.position[msg.name.index('tilt')]

    def run(self):
        r = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.reset_issued:
                # Handle reset, allow sensors to settle
                rospy.sleep(1)
                self.reset_localizer()
                self.reset_issued = False

            if self.odom:
                self.localizer.add_observation(
                    self.odom,
                    self.landmark_detections,
                    self.camera_tilt
                )
                self.localizer.optimize()
                # TODO: Add covariance?
                self.estimate_pub.publish(Pose2D(*self.localizer.get_estimated_pose()))
            r.sleep()


if __name__ == '__main__':
    print("Starting localizer node")
    listener = LocalizerListener()
    listener.run()
    rospy.spin()

        
