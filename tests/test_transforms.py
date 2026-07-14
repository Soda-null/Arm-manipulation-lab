import numpy as np

from arm_lab.kinematics.transforms import invert_transform, make_transform, rot_x, rot_y, rot_z


def test_rotation_matrices_are_orthonormal():
    for R in [rot_x(0.3), rot_y(-0.7), rot_z(1.2)]:
        assert np.allclose(R.T @ R, np.eye(3))
        assert np.isclose(np.linalg.det(R), 1.0)


def test_transform_shape_and_inverse():
    T = make_transform(rot_z(0.4), np.array([1.0, -0.2, 0.5]))
    assert T.shape == (4, 4)
    assert np.allclose(T @ invert_transform(T), np.eye(4))
