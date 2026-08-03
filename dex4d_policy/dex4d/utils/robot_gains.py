"""FR3 arm gain rows, copied verbatim from SimToolReal robots.py at ed6ea79.

kp is franka_ros2 controllers.yaml (a1, a2) or IsaacLab FRANKA_PANDA_HIGH_PD_CFG
(a3, a4). kd is either as that source ships it, or 2*sqrt(kp*armature). a3 is
what Dex4D shipped before session 032, so selecting it reproduces the old runs.

Armature is motor inertia times gear ratio squared and friction is 0.2 flat,
both from fr3v2/dynamics.yaml. They live here rather than in the URDF because
that is where SimToolReal keeps them, and its URDF still says friction zero.
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

# h1, RoboVerse's flat value. The only hand row Dex4D uses.
HAND_STIFFNESS = (3.0,) * 12
HAND_DAMPING = (0.1,) * 12
HAND_ARMATURE = (0.0,) * 12
