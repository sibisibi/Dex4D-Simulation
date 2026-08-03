"""FR3 arm gains, armature, friction, hand gains."""
# https://github.com/DAVIAN-Robotics/simtoolreal/blob/main/isaacsimenvs/tasks/simtoolreal/robots.py

ARM_STIFFNESS = (400.0,) * 7
ARM_DAMPING = (31.1307, 31.1307, 27.2029, 27.2029, 18.1328, 18.1328, 18.1328)

ARM_ARMATURE = (0.6057, 0.6057, 0.4625, 0.4625, 0.2055, 0.2055, 0.2055)
ARM_FRICTION = (0.2,) * 7

HAND_STIFFNESS = (3.0,) * 12
HAND_DAMPING = (0.1,) * 12
HAND_ARMATURE = (0.0,) * 12
