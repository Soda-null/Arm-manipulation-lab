"""Stage 04 MuJoCo model inspection.

This script loads the planar arm MuJoCo XML model and prints the key model
objects needed for reaching demos.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PROJECT_ROOT / "04_mujoco_reaching/xml/planar_2link_arm.xml"


def main() -> None:
    """Load the MuJoCo model and print a compact inspection summary."""
    print("Stage 04: inspect MuJoCo model")
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    print(f"Model: {XML_PATH.name}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}, nsite={model.nsite}")
    print("Joints:")
    for joint_id in range(model.njnt):
        print(f"  - {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)}")
    print("Actuators:")
    for actuator_id in range(model.nu):
        print(f"  - {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)}")
    print("Sites:")
    for site_id in range(model.nsite):
        print(f"  - {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)}")


if __name__ == "__main__":
    main()
