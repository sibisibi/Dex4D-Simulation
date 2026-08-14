"""Pose-only interactive HTML capture for Dex4D on IsaacLab.

Exact SimToolReal style, on his instruction. This follows
`isaacsimenvs/tasks/simtoolreal/pose_viewer.py` structure for structure, a
`gym.Wrapper` that samples one env's state tensors each step and periodically
writes a Three.js URDF viewer logged as `wandb.Html`. No cameras, no Replicator,
no viewport, same as theirs.

Three things are Dex4D's rather than SimToolReal's, and only three.

  the robot urdf   `fr3_xhand_dex4d.urdf`, served from the sibisibi/simtoolreal
                   raw base because that is where the mesh trees live and the
                   gym side URL-checks green against it
  the object urdf  Dex4D's corpus, `coacd_<scale>.urdf` per instance, resolved
                   through the env's own per-env assignment rather than through
                   procedural generation
  the table        Dex4D builds a box in code, not from a urdf, so a one-link
                   urdf is emitted from the same dims the env uses

The env resampling seed is `0xD3C4D`, the same value the Dex4D gym viewer uses,
so a lab arm and a gym arm at the same env count draw the same env sequence and
the media is comparable frame for frame.
"""
from __future__ import annotations

import base64
import os.path as osp
import random
import struct
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote

import gymnasium as gym
import numpy as np

# utils/pose_viewer.py:28. The fork is local-only, so the browser fetches robot
# meshes from a public repo carrying the identical mesh tree. The robot URDF
# itself is read off local disk and its refs rewritten, never fetched.
ROBOT_RAW_BASES = {
    "fr3_xhand_description": (
        "https://raw.githubusercontent.com/sibisibi/simtoolreal/main/"
        "assets/urdf/fr3_xhand_description/"),
}
GOAL_COLOR = (0.20, 0.72, 0.31)   # SimToolReal goal green


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _quat_wxyz_to_xyzw(quat) -> np.ndarray:
    q = _to_numpy(quat)
    return q[[1, 2, 3, 0]]


def _pose_xyzw(pos, quat_wxyz) -> np.ndarray:
    pose = np.zeros(7, dtype=np.float32)
    pose[:3] = _to_numpy(pos)
    pose[3:] = _quat_wxyz_to_xyzw(quat_wxyz)
    return pose


def _obj_to_stl_data_uri(obj_path):
    """utils/pose_viewer.py:_obj_to_stl_data_uri, verbatim.

    The template's OBJ branch derives a base URL and throws on data URIs, its
    STL branch loads them and recomputes normals, so embedded meshes travel as
    STL. This is also why an OBJ-backed goal renders white without it.
    """
    vertices, faces = [], []
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = [int(tok.split("/")[0]) for tok in parts[1:]]
                assert all(i > 0 for i in idx), f"negative OBJ indices in {obj_path}"
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0] - 1, idx[k] - 1, idx[k + 1] - 1))
    assert vertices and faces, f"no geometry parsed from {obj_path}"
    verts = np.asarray(vertices, dtype="<f4")
    blob = bytearray(struct.pack("<80xI", len(faces)))
    zero_normal = struct.pack("<3f", 0.0, 0.0, 0.0)
    for a, b, c in faces:
        blob += zero_normal
        blob += verts[a].tobytes() + verts[b].tobytes() + verts[c].tobytes()
        blob += b"\x00\x00"
    return ("data:model/stl;base64,"
            + base64.b64encode(bytes(blob)).decode("ascii") + "#ext=.stl")


def _embed_object_meshes(urdf_text, urdf_dir, strip_materials=False):
    """utils/pose_viewer.py:_embed_object_meshes, verbatim.

    strip_materials drops every <material>. urdf-loader re-applies the URDF
    material after loadMeshCb returns, which clobbers color_override, so the
    recoloured goal copy must shed its materials while the object copy keeps
    its own look. This is the reason a goal renders white.
    """
    root = ET.fromstring(urdf_text)
    if strip_materials:
        # Stripping alone is not enough. URDFLoader re-applies a material after
        # loadMeshCb returns (URDFLoader.js:558), and with no <material> present
        # it applies its own default, which overwrites color_override and is why
        # the goal rendered grey. So write GOAL_COLOR in as a real material and
        # let that re-application land on green. Verified green on the gym side.
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


def _rewrite_robot_mesh_urls(urdf_text, raw_base):
    """utils/pose_viewer.py:_rewrite_robot_mesh_urls, verbatim."""
    root = ET.fromstring(urdf_text)
    for mesh_elem in root.findall(".//mesh"):
        filename = mesh_elem.get("filename")
        if not filename or filename.startswith(("http://", "https://", "data:")):
            continue
        mesh_elem.set("filename", raw_base + quote(filename, safe="/"))
    return ET.tostring(root, encoding="unicode")


def _first_mesh_url(urdf_text):
    root = ET.fromstring(urdf_text)
    for mesh_elem in root.findall(".//mesh"):
        fn = mesh_elem.get("filename")
        if fn and fn.startswith(("http://", "https://")):
            return fn
    raise RuntimeError("robot URDF has no http(s) mesh filenames after rewrite")


def _normalize_raw_base(github_raw_base: str | None) -> str:
    base = github_raw_base or GITHUB_RAW_BASE_MAIN
    return base if base.endswith("/") else base + "/"


def _check_url(url: str, url_check: str) -> None:
    if url_check == "skip":
        return
    print(f"[pose_viewer] URL check ({url_check}) -> {url}", flush=True)
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            code = r.status
    except Exception as exc:                      # noqa: BLE001
        code = f"failed, {exc}"
    ok = code == 200
    print(f"[pose_viewer]   {'PASSED' if ok else 'FAILED'} ({code})", flush=True)
    if not ok and url_check == "error":
        raise RuntimeError(f"pose_viewer mesh base unreachable, {url} -> {code}")


def table_urdf_text(dims) -> str:
    """Dex4D builds its table with `create_box`, so the viewer needs a urdf that
    matches those dims rather than a file on disk."""
    dx, dy, dz = dims
    return (
        '<?xml version="1.0"?>\n<robot name="table">\n  <link name="box">\n'
        f'    <visual><origin xyz="0 0 0"/><geometry>'
        f'<box size="{dx} {dy} {dz}"/></geometry>'
        '<material name="wood"><color rgba="0.82 0.56 0.35 1.0"/></material></visual>\n'
        f'    <collision><origin xyz="0 0 0"/><geometry>'
        f'<box size="{dx} {dy} {dz}"/></geometry></collision>\n'
        "  </link>\n</robot>\n")


def object_urdf_for_env(env, env_id: int) -> tuple[str, Path]:
    """The corpus urdf this env was assigned at scene build.

    Dex4D fixes one object per env for the whole run, so this is stable between
    captures, the same as SimToolReal's `_object_asset_index_per_env`.
    """
    assignment = getattr(env, "_assignment", None)
    keys = getattr(env, "_cache_keys", None)
    corpus = getattr(env, "_corpus_root", None)
    if assignment is None or keys is None or corpus is None:
        raise RuntimeError(
            "Dex4D lab env does not expose the object mapping. Expected "
            "_assignment, _cache_keys and _corpus_root.")
    code, scale = keys[assignment[env_id]].split("|")
    path = Path(corpus) / code / "coacd" / f"coacd_{scale}.urdf"
    return path.read_text(encoding="utf-8"), path


def capture_pose_viewer_frame(env, env_id: int) -> dict[str, Any]:
    """One env-local frame. Mirrors SimToolReal's function of the same name."""
    if env_id < 0 or env_id >= env.num_envs:
        raise ValueError(f"env_id={env_id} out of range for num_envs={env.num_envs}")
    origin = env.scene.env_origins[env_id]

    # Canonical joint order, so the viewer's urdf names line up. IsaacLab's own
    # order is not the urdf's, which is the boundary rule the whole port keeps.
    joint_pos = env.robot.data.joint_pos[env_id, env.c2l]
    from robots import ARM_JOINTS, HAND_JOINTS
    joint_names = list(ARM_JOINTS + HAND_JOINTS)

    return {
        "env_id": int(env_id),
        "robot_joint_names": joint_names,
        "robot_joint_pos": _to_numpy(joint_pos),
        "robot_base_pose": _pose_xyzw(env.robot.data.root_pos_w[env_id] - origin,
                                      env.robot.data.root_quat_w[env_id]),
        "object_pose": _pose_xyzw(env.object.data.root_pos_w[env_id] - origin,
                                  env.object.data.root_quat_w[env_id]),
        "goal_pose": _pose_xyzw(env.goal_viz.data.root_pos_w[env_id] - origin,
                                env.goal_viz.data.root_quat_w[env_id]),
        "table_pose": _pose_xyzw(env.table.data.root_pos_w[env_id] - origin,
                                 env.table.data.root_quat_w[env_id]),
    }


def build_pose_viewer_html(*, frames, robot_urdf_text, object_urdf_text,
                           goal_urdf_text, table_urdf_text) -> str:
    """Same construction as SimToolReal's function of the same name.

    Robot, object and table urdfs are embedded and their mesh filenames
    rewritten to GitHub raw urls so the browser can fetch them. The goal is the
    same object urdf tinted green, exactly as theirs.

    The robot urdf is fetched from the raw base rather than read off disk,
    because Dex4D's copy lives on nas4 and the mesh trees the viewer needs are
    the simtoolreal ones. Its refs are already relative, verified on the gym
    side tonight, so no rewrite is needed.
    """
    from isaacsimenvs.utils.interactive_viewer import create_html, make_embedded_robot

    if not frames:
        raise ValueError("Cannot build pose viewer from zero frames.")
    timestamps = np.arange(len(frames), dtype=np.float32) / 60.0
    robots = [
        make_embedded_robot(name="robot", urdf_text=robot_urdf_text, animated=True),
        make_embedded_robot(name="table", urdf_text=table_urdf_text),
        make_embedded_robot(name="object", urdf_text=object_urdf_text),
        make_embedded_robot(name="goal", urdf_text=goal_urdf_text,
                            color_override=GOAL_COLOR),
    ]
    object_poses = {
        "table": np.stack([f["table_pose"] for f in frames]),
        "object": np.stack([f["object_pose"] for f in frames]),
        "goal": np.stack([f["goal_pose"] for f in frames]),
    }
    return create_html(
        joint_names=frames[0]["robot_joint_names"],
        robot_joint_positions=np.stack([f["robot_joint_pos"] for f in frames]),
        robots=robots,
        object_poses=object_poses,
        robot_base_poses=np.stack([f["robot_base_pose"] for f in frames]),
        timestamps=timestamps,
    )


class Dex4DPoseViewerWrapper(gym.Wrapper):
    """Periodically writes pose-only interactive HTML rollouts.

    Same shape as `SimToolRealPoseViewerWrapper`, including the separate RNG so
    viewer env sampling never perturbs the training seed, and the rebind at the
    start of every capture window so the media does not show one env forever.
    """

    def __init__(self, env, *, output_dir, capture_len: int, capture_interval: int,
                 env_id: int = 0, wandb_key: str = "interactive_viewer",
                 github_raw_base: str | None = None, url_check: str = "skip") -> None:
        super().__init__(env)
        if capture_len <= 0:
            raise ValueError(f"capture_len must be > 0, got {capture_len}")
        if url_check not in {"skip", "warn", "error"}:
            raise ValueError(f"url_check must be skip/warn/error, got {url_check}")

        inner = self.env.unwrapped
        if env_id < 0 or env_id >= inner.num_envs:
            raise ValueError(f"env_id={env_id} out of range for {inner.num_envs}")

        self.output_dir = Path(output_dir)
        self.capture_len = int(capture_len)
        self.capture_interval = int(capture_interval)
        self.wandb_key = wandb_key
        self.raw_base = _normalize_raw_base(
            github_raw_base or ROBOT_RAW_BASES["fr3_xhand_description"])
        self.url_check = url_check
        self._table_dims = inner.d4.scene.table_dims

        # Local urdf, refs rewritten. Never fetched, which is what 404'd before.
        self._robot_urdf_text = _rewrite_robot_mesh_urls(
            Path(inner.d4.robot_urdf_src).read_text(encoding="utf-8"), self.raw_base)
        _check_url(_first_mesh_url(self._robot_urdf_text), url_check)

        # Separate RNG, same seed as the Dex4D gym viewer, so a lab arm and a
        # gym arm at the same env count walk the same env sequence.
        self._env_rng = random.Random(0xD3C4D)
        self._bind_env(self._env_rng.randrange(inner.num_envs))

        self._step = 0
        self._capture_index = 0
        self._frames: list | None = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[pose_viewer] enabled: env resampled per capture, "
              f"first env_id={self.env_id} len={self.capture_len} "
              f"interval={self.capture_interval} output_dir={self.output_dir}",
              flush=True)

    def _bind_env(self, env_id: int) -> None:
        inner = self.env.unwrapped
        self.env_id = int(env_id)
        raw_text, path = object_urdf_for_env(inner, self.env_id)
        # Object keeps its own materials, goal sheds them so color_override wins.
        self._object_urdf_text = _embed_object_meshes(raw_text, str(path.parent))
        self._goal_urdf_text = _embed_object_meshes(
            raw_text, str(path.parent), strip_materials=True)
        self._table_urdf_text = table_urdf_text(self._table_dims)

    def step(self, action):
        result = self.env.step(action)
        self._step += 1

        if (self._frames is None and self.capture_interval > 0
                and self._step % self.capture_interval == 0):
            self._bind_env(self._env_rng.randrange(self.env.unwrapped.num_envs))
            self._frames = []

        if self._frames is not None:
            self._frames.append(
                capture_pose_viewer_frame(self.env.unwrapped, self.env_id))
            if len(self._frames) >= self.capture_len:
                self._finalize_capture()
        return result

    def _finalize_capture(self) -> None:
        frames = self._frames
        self._frames = None
        self._capture_index += 1
        html = build_pose_viewer_html(
            frames=frames,
            robot_urdf_text=self._robot_urdf_text,
            object_urdf_text=self._object_urdf_text,
            goal_urdf_text=self._goal_urdf_text,
            table_urdf_text=self._table_urdf_text)
        out = self.output_dir / (f"capture_{self._capture_index:04d}_"
                                 f"step{self._step}_env{self.env_id}.html")
        out.write_text(html, encoding="utf-8")
        self._log_wandb(html)
        print(f"[pose_viewer] wrote {len(frames)} frames, env_id={self.env_id} -> {out}",
              flush=True)

    def _log_wandb(self, html_text: str) -> None:
        import wandb
        if wandb.run is None:
            return
        wandb.log({self.wandb_key: wandb.Html(html_text),
                   "pose_viewer/env_id": self.env_id})

    def close(self) -> None:
        if self._frames:
            self._finalize_capture()
        return self.env.close()
