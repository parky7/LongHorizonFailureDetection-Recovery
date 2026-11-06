import unittest
import numpy as np
from simulation.oracle import check


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.state = dict(cube_pose=np.array([0., 0., .02, 1, 0, 0, 0]),
                          goal_pose=np.array([0., 0., .02, 1, 0, 0, 0]),
                          tcp_pose=np.array([0., 0., .07, 1, 0, 0, 0]),
                          is_grasped=False, success=True)

    def test_proximity_does_not_imply_grasp(self):
        self.assertTrue(check('approach_object', self.state, .02))
        self.assertFalse(check('grasp_object', self.state, .02))

    def test_lift_requires_grasp_and_height(self):
        self.state['is_grasped'] = True
        self.assertFalse(check('lift_object', self.state, .02))
        self.state['cube_pose'][2] = .12
        self.assertTrue(check('lift_object', self.state, .02))
        self.state['is_grasped'] = False
        self.assertFalse(check('lift_object', self.state, .02))

    def test_release_requires_goal_and_task_success(self):
        self.assertTrue(check('release_object', self.state, .02))
        self.state['cube_pose'][0] = .10
        self.assertFalse(check('release_object', self.state, .02))
        self.state['cube_pose'][0] = 0
        self.state['success'] = False
        self.assertFalse(check('release_object', self.state, .02))


if __name__ == '__main__':
    unittest.main()
