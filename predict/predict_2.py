import rospy
import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


class predict_2:

    # Initialize the class
    def __init__(self):

        rospy.init_node('predict_2', anonymous=True, log_level=rospy.INFO)
        self.chesses = []
        self.fingers_new = []
        self.fingers_mean = []
        self.fingers_mean_old = None
        self.v_scores = [0, 0, 0]
        self.scores = [0, 0, 0]
        self.wrist = []
        self.begin_sign = 1


    def run(self):
        while not rospy.is_shutdown():
            sign = 1
            for chess_id in [0, 1, 2]:
                if not rospy.has_param('/chess_' + str(chess_id)):
                    sign = 0
            if sign:
                for chess_id in [0, 1, 2]:
                    self.chesses.append(np.array(rospy.get_param('/chess_' + str(chess_id))))
                break
        while not rospy.is_shutdown():
            self.fingers_new = []
            # self.scores = []
            self.wrist = []
            for finger_id in range(21):
            # for finger_id in [0]:
                if rospy.has_param('/finger_' + str(finger_id)):
                    self.fingers_new.append(np.array(rospy.get_param('/finger_' + str(finger_id))))
            if rospy.has_param('/wrist'):
                self.wrist = np.array(rospy.get_param('/wrist'))
            if self.begin_sign:
                self.fingers_mean = np.mean(self.fingers_new, axis=0)
                self.begin_sign = 0
                continue

            self.fingers_mean_old = self.fingers_mean
            self.fingers_mean = np.mean(self.fingers_new, axis=0)
            v = self.fingers_mean - self.fingers_mean_old
            # print(self.fingers_mean)
            for i, chess in enumerate(self.chesses):
                # for finger in self.fingers_new:
                score = 1 / np.sum((self.fingers_mean - chess) ** 2)
                if v[0] != 0.0 and v[1] != 0.0 and v[2] != 0.0:
                    proj = normalize(chess - self.fingers_mean_old)
                    self.v_scores[i] = np.dot(proj, v) * 500 + self.v_scores[i] * 0.8
                self.scores[i] = score + self.v_scores[i]

            max_id = np.argmax(self.scores)
            max_score_norm = self.scores[max_id] / np.sum(self.scores)
            # print(self.v_scores)
            print(max_id)

if __name__ == '__main__':
    # relay_obj = predict_hand_vf()

    # rospy.spin()
    pre = predict_2()
    pre.run()
    # print(normalize([1, 1, 1]))
