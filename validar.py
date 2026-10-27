from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
import numpy as np

with open("hola.pose", "rb") as f:
    pose = Pose.read(f.read())

print("Frames:", pose.body.data.shape[0])
print("Confianza promedio:", pose.body.confidence.mean())
print("¿NaNs?:", np.isnan(pose.body.data).any())

v = PoseVisualizer(pose)
v.save_video("check_hola.mp4", v.draw())