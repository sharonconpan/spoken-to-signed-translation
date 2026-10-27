import sys
from pathlib import Path
from pose_format.pose import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import pose_normalization_info, normalize_pose_size

pose_path = Path(sys.argv[1])

with open(pose_path, "rb") as f:
    pose = Pose.read(f.read())

print("frames:", pose.body.data.shape[0], "| fps:", pose.body.fps)
conf = pose.body.confidence
frames_sin_manos = (conf[:, 0, -42:].mean(axis=1) < 0.1).sum()
print(f"frames con manos no detectadas: {frames_sin_manos}/{pose.body.data.shape[0]}")

pose = pose.normalize(pose_normalization_info(pose.header))
normalize_pose_size(pose)
v = PoseVisualizer(pose)
v.save_video("revision.mp4", v.draw())
print("listo -> revision.mp4")