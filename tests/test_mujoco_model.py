"""Tests for Stage 04 MuJoCo model loading."""

from __future__ import annotations

from pathlib import Path
import unittest

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PROJECT_ROOT / "04_mujoco_reaching/xml/planar_2link_arm.xml"


class TestMuJoCoModel(unittest.TestCase):
    """Smoke tests for the planar 2-link MuJoCo model."""

    def test_planar_arm_xml_loads(self) -> None:
        """The MuJoCo XML should load and expose the expected model sizes."""
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))

        self.assertEqual(model.nq, 2)
        self.assertEqual(model.nv, 2)
        self.assertEqual(model.nu, 2)
        self.assertEqual(model.njnt, 2)
        self.assertEqual(model.nsite, 2)

    def test_expected_names_exist(self) -> None:
        """The model should contain named joints, actuators, and sites used by demos."""
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))

        for object_type, name in [
            (mujoco.mjtObj.mjOBJ_JOINT, "shoulder"),
            (mujoco.mjtObj.mjOBJ_JOINT, "elbow"),
            (mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_motor"),
            (mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_motor"),
            (mujoco.mjtObj.mjOBJ_SITE, "target"),
            (mujoco.mjtObj.mjOBJ_SITE, "end_effector"),
        ]:
            with self.subTest(name=name):
                self.assertGreaterEqual(mujoco.mj_name2id(model, object_type, name), 0)


if __name__ == "__main__":
    unittest.main()
