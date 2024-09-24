import rospy
import numpy as np
# from prediction_msgs.msg import Prediction
from trajectory_msgs.msg import JointTrajectoryPoint

def handtraj_cb(self, handtraj):
    print(handtraj)

rospy.init_node('predict_hand_vf', anonymous=True, log_level=rospy.DEBUG)
handtraj_sub = rospy.Subscriber('/hand_traj', JointTrajectoryPoint, handtraj_cb)
rospy.spin()


