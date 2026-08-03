"""Pose-only interactive HTML capture for Dex4D training.

Port of the SimToolReal pose viewer (simtoolreal/isaacsimenvs/tasks/
simtoolreal/pose_viewer.py). Samples one env's state tensors each control
step and periodically writes a Three.js/URDF HTML viewer that is saved to
the logdir and logged to wandb as wandb.Html. No cameras, no renderer.
"""

from __future__ import annotations

import base64
import os.path as osp
import random
import struct
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import numpy as np
import wandb

from utils.interactive_viewer import create_html, make_embedded_robot

# This fork is local-only, so the browser fetches robot meshes from public
# repos that carry identical mesh trees (all HEAD-checked 200 on 2026-07-16).
ROBOT_RAW_BASES = {
    "xarm6_leap_description": (
        "https://raw.githubusercontent.com/Dex4D/Dex4D-Simulation/master/"
        "dex4d_policy/assets/urdf/xarm6_leap_description/"
    ),
    "fr3_xhand_description": (
        "https://raw.githubusercontent.com/sibisibi/simtoolreal/main/"
        "assets/urdf/fr3_xhand_description/"
    ),
}

GOAL_COLOR = (0.20, 0.72, 0.31)  # SimToolReal goal green

TABLE_URDF = """<robot name="table">
  <link name="table_top">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{x:.4f} {y:.4f} {z:.4f}"/></geometry>
      <material name="table_gray"><color rgba="0.59 0.59 0.59 1"/></material>
    </visual>
  </link>
</robot>
"""


def _obj_to_stl_data_uri(obj_path):
    """Convert a triangulated OBJ mesh into a binary-STL data URI.

    The viewer template's OBJ branch derives a base URL from the mesh URL,
    which throws on data URIs, while its STL branch loads them fine and
    recomputes normals, so embedded meshes travel as STL. The trailing
    #ext=.stl marker is the template's format hint for extension-less URLs.
    """
    vertices = []
    faces = []
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = [int(tok.split("/")[0]) for tok in parts[1:]]
                assert all(i > 0 for i in idx), "negative OBJ indices in {}".format(obj_path)
                # fan triangulation, exact for the convex CoACD pieces
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0] - 1, idx[k] - 1, idx[k + 1] - 1))
    assert vertices and faces, "no geometry parsed from {}".format(obj_path)
    verts = np.asarray(vertices, dtype="<f4")
    blob = bytearray(struct.pack("<80xI", len(faces)))
    zero_normal = struct.pack("<3f", 0.0, 0.0, 0.0)
    for a, b, c in faces:
        blob += zero_normal
        blob += verts[a].tobytes() + verts[b].tobytes() + verts[c].tobytes()
        blob += b"\x00\x00"
    payload = base64.b64encode(bytes(blob)).decode("ascii")
    return "data:model/stl;base64," + payload + "#ext=.stl"


def _rewrite_robot_mesh_urls(urdf_text, raw_base):
    """Point every relative mesh filename at the public raw-GitHub tree."""
    root = ET.fromstring(urdf_text)
    for mesh_elem in root.findall(".//mesh"):
        filename = mesh_elem.get("filename")
        if not filename or filename.startswith(("http://", "https://", "data:")):
            continue
        mesh_elem.set("filename", raw_base + quote(filename, safe="/"))
    return ET.tostring(root, encoding="unicode")


def _embed_object_meshes(urdf_text, urdf_dir, strip_materials=False):
    """Inline the object's OBJ meshes as STL data URIs (dataset is local-only).

    strip_materials drops every URDF <material> element. urdf-loader re-applies
    the URDF material to a mesh after loadMeshCb returns (URDFLoader.js:558),
    which clobbers color_override, so the recolored goal copy must shed its
    materials while the object copy keeps its own look.
    """
    root = ET.fromstring(urdf_text)
    if strip_materials:
        # Stripping alone is not enough. URDFLoader re-applies a material after
        # loadMeshCb returns (URDFLoader.js:558), and with no <material> present
        # it applies its own default, which overwrites color_override and is why
        # the goal rendered grey. So write GOAL_COLOR in as a real material and
        # let that re-application land on green.
        r, g, b = GOAL_COLOR
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "material":
                    parent.remove(child)
        for visual in root.findall(".//visual"):
            mat = ET.SubElement(visual, "material")
            mat.set("name", "goal_green")
            col = ET.SubElement(mat, "color")
            col.set("rgba", "{} {} {} 1".format(r, g, b))
    cache = {}
    for mesh_elem in root.findall(".//mesh"):
        filename = mesh_elem.get("filename")
        if not filename or filename.startswith(("http://", "https://", "data:")):
            continue
        mesh_path = osp.normpath(osp.join(urdf_dir, filename))
        if mesh_path not in cache:
            cache[mesh_path] = _obj_to_stl_data_uri(mesh_path)
        mesh_elem.set("filename", cache[mesh_path])
    return ET.tostring(root, encoding="unicode")


def _first_mesh_url(urdf_text):
    root = ET.fromstring(urdf_text)
    for mesh_elem in root.findall(".//mesh"):
        filename = mesh_elem.get("filename")
        if filename and filename.startswith(("http://", "https://")):
            return filename
    raise RuntimeError("robot URDF has no http(s) mesh filenames after rewrite")


def _check_url(url, url_check):
    if url_check == "skip":
        return
    print("[pose_viewer] URL check ({}) -> {}".format(url_check, url), flush=True)
    start = time.monotonic()
    try:
        request = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(request, timeout=10)
        print("[pose_viewer]   PASSED ({:.2f}s)".format(time.monotonic() - start), flush=True)
    except Exception as exc:
        message = "[pose_viewer]   FAILED ({:.2f}s): {}".format(time.monotonic() - start, exc)
        if url_check == "error":
            raise RuntimeError(message) from exc
        print(message, flush=True)


def _pose7(root_state_row, origin):
    pose = root_state_row[:7].detach().cpu().numpy().astype(np.float32).copy()
    pose[:3] -= origin
    return pose  # Isaac Gym root-state quats are already xyzw


class Dex4DPoseViewer:
    """Samples env `env_id` every control step. Every `capture_interval`
    steps it collects `capture_len` frames, writes an interactive HTML into
    `output_dir`, and logs it to the live wandb run under `wandb_key`.
    The first capture starts at step 1 so media lands early in the run.
    """

    def __init__(
        self,
        task,
        output_dir,
        capture_len=600,
        capture_interval=1000,
        env_id=0,
        wandb_key="interactive_viewer",
        robot_raw_base="",
        url_check="skip",
    ):
        if capture_len <= 0:
            raise ValueError("capture_viewer_len must be > 0, got {}".format(capture_len))
        if url_check not in ("skip", "warn", "error"):
            raise ValueError("capture_viewer_url_check must be skip/warn/error, got {}".format(url_check))
        if env_id < 0 or env_id >= task.num_envs:
            raise ValueError("capture_viewer_env_id={} out of range for num_envs={}".format(env_id, task.num_envs))

        self.task = task
        self.output_dir = Path(output_dir)
        self.capture_len = int(capture_len)
        self.capture_interval = int(capture_interval)
        self.env_id = int(env_id)
        self.wandb_key = wandb_key
        # independent RNG so env resampling never perturbs the training seed
        self._env_rng = random.Random(0xD3C4D)

        # robot URDF, embedded text with meshes rewritten to public raw URLs
        asset_root = task.cfg["env"]["asset"].get("assetRoot", "../../assets")
        asset_file = task.cfg["env"]["asset"]["assetFileName"]
        robot_urdf_path = Path(osp.realpath(osp.join(asset_root, asset_file)))
        if not robot_raw_base:
            robot_raw_base = ROBOT_RAW_BASES[robot_urdf_path.parent.name]
        if not robot_raw_base.endswith("/"):
            robot_raw_base += "/"
        self._robot_urdf_text = _rewrite_robot_mesh_urls(
            robot_urdf_path.read_text(encoding="utf-8"), robot_raw_base
        )
        _check_url(_first_mesh_url(self._robot_urdf_text), url_check)

        # table URDF from the sim's box dims
        dims = task.table_dims
        self._table_urdf_text = TABLE_URDF.format(x=dims.x, y=dims.y, z=dims.z)

        # embodiment-wide constants, identical across envs
        self._joint_names = list(task.gym.get_actor_dof_names(task.envs[0], task.robots[0]))
        assert len(self._joint_names) == task.robot_dof_pos.shape[1], (
            "actor dof names ({}) do not match robot_dof_pos width ({})".format(
                len(self._joint_names), task.robot_dof_pos.shape[1]
            )
        )
        self._dt = float(task.sim_params.dt) * int(task.cfg["env"].get("controlFrequencyInv", 1))

        self._step = 0
        self._capture_index = 0
        self._bind_env(self._env_rng.randrange(task.num_envs))
        self._frames = []  # capture from step 1, first HTML lands at step capture_len

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[pose_viewer] enabled: env resampled per capture, first env_id={} len={} interval={} object={}/{} output_dir={}".format(
                self.env_id, self.capture_len, self.capture_interval,
                self._code_scale["object_code"], self._code_scale["scale_str"], self.output_dir,
            ),
            flush=True,
        )

    def _bind_env(self, env_id):
        """Bind every per-env constant. Called at construction and again at the
        start of each capture window, so successive captures show different
        envs and their assigned objects."""
        task = self.task
        self.env_id = int(env_id)

        # object and goal URDF, this env's assigned object with meshes inlined
        code_scale = task.object_code_and_scale_str_for_envs[self.env_id]
        self._code_scale = code_scale
        object_urdf_path = Path(osp.realpath(osp.join(
            "../assets/meshdatav3_scaled",
            code_scale["object_code"],
            "coacd",
            "coacd_{}.urdf".format(code_scale["scale_str"]),
        )))
        object_urdf_raw = object_urdf_path.read_text(encoding="utf-8")
        self._object_urdf_text = _embed_object_meshes(object_urdf_raw, str(object_urdf_path.parent))
        self._goal_urdf_text = _embed_object_meshes(
            object_urdf_raw, str(object_urdf_path.parent), strip_materials=True
        )

        # This codebase keeps root_state_tensor values env-local (resets write the
        # raw create_actor poses back, UniDexGrasp lineage), so NO origin subtraction.
        # The SimToolReal original subtracts get_env_origin because IsaacLab root
        # states are world-frame; the conventions differ, do not re-add it here.
        self._origin = np.zeros(3, dtype=np.float32)
        self._robot_index = int(task.robot_indices[self.env_id])
        self._object_index = int(task.object_indices[self.env_id])
        self._goal_index = int(task.goal_object_indices[self.env_id])
        self._table_index = int(task.table_indices[self.env_id])

    def on_step(self):
        self._step += 1
        if self._frames is None and self.capture_interval > 0 and self._step % self.capture_interval == 0:
            self._bind_env(self._env_rng.randrange(self.task.num_envs))
            self._frames = []
        if self._frames is not None:
            self._frames.append(self._capture_frame())
            if len(self._frames) >= self.capture_len:
                self._finalize()

    def _capture_frame(self):
        task = self.task
        root_states = task.root_state_tensor
        return {
            "robot_joint_pos": task.robot_dof_pos[self.env_id].detach().cpu().numpy().astype(np.float32).copy(),
            "robot_base_pose": _pose7(root_states[self._robot_index], self._origin),
            "object_pose": _pose7(root_states[self._object_index], self._origin),
            "goal_pose": _pose7(root_states[self._goal_index], self._origin),
            "table_pose": _pose7(root_states[self._table_index], self._origin),
        }

    def _finalize(self):
        frames = self._frames
        self._frames = None

        robots = [
            make_embedded_robot(name="robot", urdf_text=self._robot_urdf_text, animated=True),
            make_embedded_robot(name="table", urdf_text=self._table_urdf_text),
            make_embedded_robot(name="object", urdf_text=self._object_urdf_text),
            make_embedded_robot(name="goal", urdf_text=self._goal_urdf_text, color_override=GOAL_COLOR),
        ]
        html_text = create_html(
            joint_names=self._joint_names,
            robot_joint_positions=np.stack([f["robot_joint_pos"] for f in frames]),
            robots=robots,
            object_poses={
                "table": np.stack([f["table_pose"] for f in frames]),
                "object": np.stack([f["object_pose"] for f in frames]),
                "goal": np.stack([f["goal_pose"] for f in frames]),
            },
            robot_base_poses=np.stack([f["robot_base_pose"] for f in frames]),
            dt=self._dt,
        )
        html_path = self.output_dir / "pose_viewer_step_{:09d}_{:04d}_env{:05d}.html".format(
            self._step, self._capture_index, self.env_id
        )
        html_path.write_text(html_text, encoding="utf-8")
        print("[pose_viewer] wrote {} frames (env {}, object {}/{}) to {}".format(
            len(frames), self.env_id, self._code_scale["object_code"],
            self._code_scale["scale_str"], html_path), flush=True)

        if wandb.run is not None:
            wandb.log({self.wandb_key: wandb.Html(html_text), "pose_viewer/env_id": self.env_id})
            wandb.run.summary["interactive_viewer_latest"] = wandb.Html(html_text)
            print("[pose_viewer] logged wandb Html key={} step={} env={}".format(
                self.wandb_key, self._step, self.env_id), flush=True)

        self._capture_index += 1
