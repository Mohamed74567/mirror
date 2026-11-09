"""Mirror — closed-loop monocular video to MuJoCo digital-twin pipeline.

Package layout
--------------
``src.vision_extractor``  Part 1 — video ingestion, keypoint tracking, kinematics
``src.mjcf_builder``      Part 2 — parameterized dynamic MJCF compilation
``src.mujoco_sim``        Part 3 — forward-dynamics simulation + silhouettes
``src.optimizer``         Part 4 — closed-loop silhouette-IoU calibration
"""

__version__ = "1.1.0"

__all__ = ["vision_extractor", "mjcf_builder", "mujoco_sim", "optimizer", "__version__"]
