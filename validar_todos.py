# validar_todos.py
from pathlib import Path
from pose_format.pose import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import pose_normalization_info, normalize_pose_size

carpeta = Path("assets/lexicon_lsp/prl")
salida = Path("revisiones")
salida.mkdir(exist_ok=True)

resultados = []

for pose_path in sorted(carpeta.glob("*.pose")):
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    total_frames = pose.body.data.shape[0]
    conf = pose.body.confidence
    frames_sin_manos = int((conf[:, 0, -42:].mean(axis=1) < 0.1).sum())
    pct = frames_sin_manos / total_frames * 100 if total_frames else 0
    resultados.append((pose_path.stem, total_frames, frames_sin_manos, pct))

    pose_norm = pose.normalize(pose_normalization_info(pose.header))
    normalize_pose_size(pose_norm)
    v = PoseVisualizer(pose_norm)
    v.save_video(str(salida / f"{pose_path.stem}.mp4"), v.draw())

print(f"\n{'palabra':<12}{'frames':<10}{'sin manos':<12}{'%':<8}")
for nombre, total, sin_manos, pct in resultados:
    alerta = "  <-- REVISAR" if pct > 20 else ""
    print(f"{nombre:<12}{total:<10}{sin_manos:<12}{pct:<8.1f}{alerta}")