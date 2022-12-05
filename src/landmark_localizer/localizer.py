import math
import gtsam
import numpy as np
from gtsam import symbol_shorthand

L = symbol_shorthand.L
X = symbol_shorthand.X

# Create noise models
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.01, 0.01, 0.01], dtype=float))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.01, 0.01, math.pi/32], dtype=float))
LANDMARK_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.5, 0.5], dtype=float))


def calc_pos_from_bearing_range(pose, l_bearing, l_range):
    '''Calculate the global coordinate of landmark based on 
    current pose, bearing, and range'''
    # Calculate relative x and y from robot pose
    dx = l_range * np.cos(l_bearing)
    dy = l_range * np.sin(l_bearing)

    # Rotate opposite robot pose to get global change in x and y from pose
    # Add to robot's pose coordinates
    r_x, r_y, r_theta = pose
    l_x = r_x + (dx * np.cos(-r_theta) + dy * np.sin(-r_theta))
    l_y = r_y + (dx * np.sin(r_theta) + dy * np.cos((-r_theta)))

    return (l_x, l_y)


def calc_bearing_range_from_tag(tag, camera_tilt=0):
    '''Takes AprilTag detection data and calculates a bearing and range,
    taking into account the tilt of the camera.'''
    # Tag data (in meters)
    # z = distance to center of tag to center of camera
    # x = real horizontal distance from camera center, left relative to robot is postive
    # y = real vertical distance from camera center, down is positive

    # Camera is 0.03m from true center
    camera_x_offset = 0.03

    shifted_x = tag.x - camera_x_offset

    # Project the tag distance to the camera center parallel to the ground based on camera tilt
    # dist_to_camera_center = math.sqrt(tag.z**2 - shifted_x**2 - tag.y**2) * np.cos(camera_tilt)
    
    ground_projection = math.sqrt(tag.z**2 - tag.y**2) * np.cos(camera_tilt) + (-tag.y * np.sin(camera_tilt))
    print(f"Ground projection: {ground_projection}")

    dist_to_camera_center = math.sqrt(ground_projection**2 - shifted_x**2)
    print(f"Dist camera center: {dist_to_camera_center}")

    # Adjust for placement of camera from center of LoCoBot
    camera_x_dist = 0.07
    t_range = math.sqrt(shifted_x**2 + (dist_to_camera_center + camera_x_dist)**2)

    # Find the bearing angle from LoCobot center to tag
    t_bearing = math.asin(shifted_x / t_range)

    # Flip bearing for correct bot rotation direction
    return -t_bearing, t_range


class BlockBotLocalizer:
    def __init__(self, start=(0, 0, 0), use_landmarks=True, verbose=True) -> None:
        self.v = verbose

        # Track pose id index
        self.current_idx = 0
        self.use_landmarks = use_landmarks
        self.debug = True

        # Track odom history and seen landmarks
        self.seen_landmarks = set()
        self.odom_history = [start]

        # Init the factor graph
        self.initial_estimate = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()

        # Set the starting pose as a PriorFactor
        start_pose = gtsam.Pose2(*start)

        self.graph.add(gtsam.PriorFactorPose2(X(0), start_pose, PRIOR_NOISE))
        self.initial_estimate.insert(X(self.current_idx), start_pose)
        self.optimize()

    def add_observation(self, odom, landmarks=[], camera_tilt=0.0, time_idx=None):
        '''Adds a new observation. Increments time index unless one is provided'''
        if time_idx is None:
            self.current_idx += 1
            time_idx = self.current_idx

        # Add odometry measurement
        print(f"Odometry measurement: {odom}")
        # Retreive previous odometry measurement
        prev_odom = self.odom_history[-1]
        self.odom_history.append(odom)

        # Find the change in global coordinates and pose
        dx, dy, dtheta = np.subtract(odom, prev_odom)

        # Rotate to find the change relative to the previous pose
        prev_theta = prev_odom[2]
        rel_x = dx * np.cos(prev_theta) + dy * np.sin(prev_theta)
        rel_y = dx * np.sin(-prev_theta) + dy * np.cos(prev_theta)

        # Add to factor graph and set initial estimate for current Pose
        odom_rel_pose = gtsam.Pose2(rel_x, rel_y, dtheta)
        if self.v:
            print(f'Rel pose: {odom_rel_pose}')
        self.graph.add(gtsam.BetweenFactorPose2(X(time_idx - 1), X(time_idx),
                       odom_rel_pose, ODOMETRY_NOISE))

        self.initial_estimate.insert(
            X(time_idx), gtsam.Pose2(odom[0], odom[1], odom[2]))

        # Add landmark observations
        if self.use_landmarks:
            print(f"Camera tilt: {camera_tilt}")
            for l in landmarks:
                l_id = l.id[0]

                # Get the tag data and calculate bearing and range relative to Robot pose
                tag = l.pose.pose.pose.position

                l_bearing, l_range = calc_bearing_range_from_tag(tag, camera_tilt)

                # Add BearingFactor for landmark observation from current pose
                self.graph.add(gtsam.BearingRangeFactor2D(X(time_idx), L(l_id),
                               gtsam.Rot2(l_bearing), l_range, LANDMARK_NOISE))

                if self.v:
                    print(f'Landmark {l_id} range: {l_range} bearing: {l_bearing}')
                    print(f'Raw z: {tag.z},x: {tag.x} y: {tag.y}')

                # Add estimate for landmark if not seen before
                if l_id not in self.seen_landmarks:
                    # Estimate global landmark position based on current odometry 
                    # and landmark bearing and range
                    l_pose = calc_pos_from_bearing_range(odom, l_bearing, l_range)
                    self.initial_estimate.insert(L(l_id), gtsam.Point2(*l_pose))
                    if self.v:
                        print(f'Initial estimate for {l_id}: {l_pose}')
                    self.seen_landmarks.add(l_id)

    def optimize(self):
        # Parameters for the optimization
        parameters = gtsam.GaussNewtonParams()
        parameters.setRelativeErrorTol(1e-5)
        parameters.setMaxIterations(1000)
        optimizer = gtsam.GaussNewtonOptimizer(
                    self.graph, self.initial_estimate, parameters)
        self.result = optimizer.optimize()
        marginals = gtsam.Marginals(self.graph, self.result)

        # Update estimated pose
        self.estimated_pose = self.result.atPose2(X(self.current_idx))
        self.current_covariance = marginals.marginalCovariance(X(self.current_idx))

        print("Covariance:")
        print(self.current_covariance)
        print("Estimated pose: ")
        print(self.estimated_pose)

        if self.v:
            print("Estimated landmark positions:")
            for l_id in self.seen_landmarks:
                print(f"{l_id}: {self.result.atPoint2(L(l_id))}")

    def get_estimated_pose(self):
        pose = self.estimated_pose
        return [
            pose.x(),
            pose.y(),
            pose.theta()
        ]
