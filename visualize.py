from pose_format.pose import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import pose_normalization_info, normalize_pose_size

with open('quick_test.pose', 'rb') as f:
    pose = Pose.read(f.read())

pose = pose.normalize(pose_normalization_info(pose.header))
normalize_pose_size(pose)

v = PoseVisualizer(pose)
v.save_video('quick_test_visual.mp4', v.draw())
print('Listo: quick_test_visual.mp4')