"""FR3 arm gain rows and the vendor physics set, copied from Dex4D's own
`utils/robot_gains.py`.

Copied rather than imported. Importing anything under `dex4d/` pulls
`utils/torch_jit_utils.py`, which does `from isaacgym.torch_utils import *`, and
isaacgym cannot coexist with Isaac Sim in one process. SimToolReal has exactly
one cross-reference in its whole tree for the same reason and it loads that file
by path with a comment saying so.
"""

ARM_GAIN_ROWS = {
    "a1": ((600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0),
           (30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0)),
    "a2": ((600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0),
           (38.1272, 38.1272, 33.3167, 33.3167, 14.3353, 11.1041, 6.4109)),
    "a3": ((400.0,) * 7, (80.0,) * 7),
    "a4": ((400.0,) * 7,
           (31.1307, 31.1307, 27.2029, 27.2029, 18.1328, 18.1328, 18.1328)),
}

ARM_ARMATURE = (0.6057, 0.6057, 0.4625, 0.4625, 0.2055, 0.2055, 0.2055)
ARM_FRICTION = (0.2,) * 7

HAND_STIFFNESS = (3.0,) * 12
HAND_DAMPING = (0.1,) * 12
HAND_ARMATURE = (0.0,) * 12

ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
HAND_JOINTS = (
    "thumb_joint0", "thumb_joint1", "thumb_joint2",
    "index_joint0", "index_joint1", "index_joint2",
    "middle_joint0", "middle_joint1",
    "ring_joint0", "ring_joint1",
    "pinky_joint0", "pinky_joint1",
)

# cfg/fr3_xhand_ap2ap_stage_1_2.yaml:31. SimToolReal's
# arm_default_pos, joint1 at 0 because Dex4D's base already faces the table.
RESET_ARM_POS = (0.0, 0.2618, 0.0, -2.0071, 0.0, 2.3562, -0.7854)
RESET_HAND_POS = (0.0,) * 12

# tasks/fr3_xhand_ap2ap.py:568.
BASE_POS = (-0.48, 0.0, 0.60)
TABLE_DIMS = (1.2, 1.2, 0.6)

# The five `*_tip` links are fixed children of the `*_link2` bodies and
# `merge_fixed_joints` deletes them, so the tip points are reconstructed the same
# way the palm is. Gym's `body_names` at :609 maps index/middle/ring/thumb/pinky
# onto the tips, and `obj_finger_dist`, `fingertip_pos` and the
# fingertip-to-object vectors all read them. Values are the URDF joint origins,
# and they match SimToolReal's own `fingertip_offset_by_body` for this robot
# digit for digit.
FINGERTIP_TIP_OFFSETS = {
    "index_rota_link2": (0.0, 0.0, 0.0422482924089424),
    "mid_link2": (0.0, 0.0, 0.042248),
    "ring_link2": (0.0, 0.0, 0.0422482924089404),
    "thumb_rota_link2": (0.0, 0.0502276499414863, 0.0),
    "pinky_link2": (0.0, 0.0, 0.0422482924089405),
}
