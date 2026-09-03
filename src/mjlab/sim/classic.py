"""Classic MuJoCo (C engine, CPU) simulation backend on mjbatch.

Implements the same interface as :class:`mjlab.sim.sim.Simulation`, with the physics
stepped by `mjbatch <https://github.com/kevinzakka/mjbatch>`_: N simulations on a C++
thread pool with the GIL released, each held as its ``mjSTATE_INTEGRATION`` vector and
loaded into a per-thread ``mjData`` per call, so memory scales with threads rather than
environments. Consumers see ``sim.model`` / ``sim.data`` bridges exposing batched float32
torch tensors with the same field names and shapes as the Warp bridges, so the
entity/manager/sensor stack runs unchanged.

Data fields are mjbatch ``bind`` views: torch tensors sharing memory with the batch's
arrays. A write lands element-wise at the environment's next physics call, and only what
was written lands, so untouched environments follow a plain ``mj_step`` loop bit for bit;
every bound field is refreshed after each call. Derived fields are one substep behind
``qpos``/``qvel`` after ``step()``, as with the Warp backend, and ``forward()`` refreshes
them. Model fields expanded for domain randomization are mjbatch ``expand`` views, applied
per environment before each call; ``recompute_constants`` is ``set_const``.

Not supported (Warp-only): camera/raycast sensors (``SensorContext``), mesh variants,
models with sleep enabled, and the ``wp_model``/``wp_data`` escape hatches used by the
differential-IK action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import mujoco
import numpy as np
import torch
from mjbatch import Batch

from mjlab.utils.nan_guard import NanGuard

if TYPE_CHECKING:
  import mujoco_warp as mjwarp

  from mjlab.entity.variants import VariantMetadata
  from mjlab.managers.event_manager import RecomputeLevel
  from mjlab.sensor.sensor_context import SensorContext
  from mjlab.sim.sim import SimulationCfg

# Data fields presented as (nworld, n, 3, 3) instead of MjData's (n, 9).
_DATA_MAT_FIELDS = frozenset({"xmat", "ximat", "geom_xmat", "site_xmat", "cam_xmat"})
# Model fields with a display shape differing from MjModel's layout.
_MODEL_RESHAPE = {"geom_aabb": (2, 3)}

# Bound at construction: everything the entity, sensor and manager stack reads or
# writes. A field bound later is only filled by the next physics call, so binding
# up front keeps first reads correct.
_EAGER_FIELDS = (
  "time",
  "qpos",
  "qvel",
  "qacc",
  "qacc_warmstart",
  "act",
  "ctrl",
  "xfrc_applied",
  "mocap_pos",
  "mocap_quat",
  "xpos",
  "xquat",
  "xipos",
  "xmat",
  "subtree_com",
  "cvel",
  "geom_xpos",
  "geom_xmat",
  "site_xpos",
  "site_xmat",
  "ten_length",
  "ten_velocity",
  "actuator_force",
  "qfrc_actuator",
  "sensordata",
)


def _is_num(template: Any, name: str) -> bool:
  """Whether an mjModel/mjData field is mjtNum-valued (presented as float32)."""
  val = getattr(template, name)
  return isinstance(val, float) or (
    isinstance(val, np.ndarray) and val.dtype == np.float64
  )


class ClassicData:
  """Batched torch views over the batch's ``mjData`` fields, like the Warp Data bridge."""

  _batch: Batch
  _template: mujoco.MjData
  _views: dict[str, torch.Tensor]
  nworld: int

  def __init__(self, batch: Batch, template: mujoco.MjData) -> None:
    self._batch = batch
    self._template = template
    self._views = {}
    self.nworld = batch.num_sims
    for name in _EAGER_FIELDS:
      self._bind(name)

  def _bind(self, name: str) -> torch.Tensor:
    dtype = np.float32 if _is_num(self._template, name) else None
    arr = self._batch.bind(name, dtype)
    view = torch.from_numpy(arr)
    if name in _DATA_MAT_FIELDS:
      view = view.view(*arr.shape[:-1], 3, 3)
    self._views[name] = view
    return view

  def __getattr__(self, name: str) -> torch.Tensor:
    if name.startswith("_"):
      raise AttributeError(name)
    view = self._views.get(name)
    if view is None:
      # A late-bound field holds zeros until a physics call fills it.
      view = self._bind(name)
      self._batch.forward()
    return view


class ClassicModel:
  """Batched torch views over a template ``MjModel``, like the Warp Model bridge.

  Float-valued fields present with a leading ``nworld`` dim: a broadcast view of the
  template until :meth:`expand` gives them per-environment storage in the batch.
  Integer/structural fields present unbatched. Non-array attributes (``nq``, ``opt``,
  ...) delegate to the template.
  """

  _template: mujoco.MjModel
  _batch: Batch
  _nworld: int
  _expanded: dict[str, torch.Tensor]
  _cache: dict[str, torch.Tensor]

  def __init__(self, template: mujoco.MjModel, batch: Batch) -> None:
    self._template = template
    self._batch = batch
    self._nworld = batch.num_sims
    self._expanded = {}
    self._cache = {}

  def _display(self, name: str, val: np.ndarray) -> tuple[int, ...]:
    if name in _MODEL_RESHAPE:
      return val.shape[:-1] + _MODEL_RESHAPE[name]
    return val.shape

  def expand(self, fields: tuple[str, ...]) -> None:
    """Give fields per-environment storage in the batch, seeded from the template."""
    for name in fields:
      if name in self._expanded:
        continue
      val = getattr(self._template, name)
      if not isinstance(val, np.ndarray):
        raise ValueError(f"Cannot expand non-array model field '{name}'.")
      arr = self._batch.expand(name, np.float32 if val.dtype == np.float64 else None)
      view = torch.from_numpy(arr)
      if name in _MODEL_RESHAPE:
        view = view.view(*arr.shape[:-1], *_MODEL_RESHAPE[name])
      self._expanded[name] = view
      self._cache.pop(name, None)

  def __getattr__(self, name: str) -> Any:
    if name.startswith("_"):
      raise AttributeError(name)
    if name in self._expanded:
      return self._expanded[name]
    if name in self._cache:
      return self._cache[name]
    val = getattr(self._template, name)
    if isinstance(val, np.ndarray) and val.ndim:
      t = torch.as_tensor(val.reshape(self._display(name, val)))
      if val.dtype == np.float64:
        t = t.to(torch.float32)
        t = t.unsqueeze(0).expand(self._nworld, *t.shape)
      else:
        t = t.clone()
      self._cache[name] = t
      return t
    return val


class ClassicSimulation:
  """CPU simulation with the classic MuJoCo C engine; drop-in for ``Simulation``."""

  def __init__(
    self,
    num_envs: int,
    cfg: "SimulationCfg",
    model: mujoco.MjModel | None = None,
    device: str = "cpu",
    *,
    spec: mujoco.MjSpec | None = None,
    variant_info: list[tuple[str, "VariantMetadata"]] | None = None,
  ):
    if variant_info:
      raise NotImplementedError(
        "Mesh variants are not supported by the classic backend; use backend='warp'."
      )
    if torch.device(device).type != "cpu":
      raise ValueError(f"The classic backend runs on CPU, got device='{device}'.")
    self.cfg = cfg
    self.device = device
    self.num_envs = num_envs

    if spec is not None:
      compiled: mujoco.MjModel = spec.compile()
    elif model is not None:
      compiled = model
    else:
      raise ValueError("Either model or spec must be provided.")
    cfg.mujoco.apply(compiled)

    self._mj_model: mujoco.MjModel = compiled
    self._mj_data = mujoco.MjData(compiled)
    mujoco.mj_forward(self._mj_model, self._mj_data)

    try:
      self._batch = Batch(compiled, num_sims=num_envs, num_threads=cfg.num_threads or 0)
    except ValueError as e:
      raise ValueError(f"The classic backend cannot batch this model: {e}") from e
    self.num_threads = self._batch.num_threads

    self._data_bridge = ClassicData(self._batch, self._mj_data)
    self._model_bridge = ClassicModel(compiled, self._batch)
    self._default_model_fields: dict[str, torch.Tensor] = {}
    self._expanded_fields: set[str] = set()
    self._sensor_context = None
    # Derived fields are valid from the start, as on the Warp backend.
    self._batch.forward()

    self.use_cuda_graph = False
    self.nan_guard = NanGuard(cfg.nan_guard, num_envs, compiled)

  # Properties (parity with Simulation).

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mj_data(self) -> mujoco.MjData:
    return self._mj_data

  @property
  def batch(self) -> Batch:
    """The underlying mjbatch ``Batch``."""
    return self._batch

  @property
  def data(self) -> ClassicData:
    return self._data_bridge

  @property
  def model(self) -> ClassicModel:
    return self._model_bridge

  @property
  def default_model_fields(self) -> dict[str, torch.Tensor]:
    return self._default_model_fields

  @property
  def expanded_fields(self) -> set[str]:
    return self._expanded_fields

  @property
  def per_world_default_fields(self) -> set[str]:
    return set()

  @property
  def world_to_variant(self) -> dict[str, torch.Tensor]:
    return {}

  @property
  def wp_model(self):
    raise NotImplementedError(
      "wp_model is Warp-only (used by the differential-IK action); use backend='warp'."
    )

  @property
  def wp_data(self):
    raise NotImplementedError(
      "wp_data is Warp-only (used by the differential-IK action); use backend='warp'."
    )

  @property
  def wp_device(self):
    raise NotImplementedError("wp_device is Warp-only; use backend='warp'.")

  # Methods.

  def create_graph(self) -> None:
    pass

  def expand_model_fields(self, fields: tuple[str, ...]) -> None:
    if not fields:
      return
    invalid_fields = [f for f in fields if not hasattr(self._mj_model, f)]
    if invalid_fields:
      raise ValueError(f"Fields not found in model: {invalid_fields}")
    self._model_bridge.expand(fields)
    self._expanded_fields.update(fields)

  def get_default_field(self, field: str) -> torch.Tensor:
    if field not in self._default_model_fields:
      if not hasattr(self._mj_model, field):
        raise ValueError(f"Field '{field}' not found in model")
      model_field = getattr(self.model, field)
      default_value = np.asarray(getattr(self._mj_model, field))
      display = self._model_bridge._display(field, default_value)
      self._default_model_fields[field] = torch.as_tensor(
        np.ascontiguousarray(default_value).reshape(display),
        dtype=model_field.dtype,
      ).clone()
    return self._default_model_fields[field]

  def recompute_constants(self, level: "RecomputeLevel") -> None:
    del level  # mj_setConst covers every recompute level.
    self._batch.set_const()

  def forward(self) -> None:
    self._batch.forward()

  def step(self) -> None:
    # NanGuard only does torch ops on batched fields; the bridge satisfies it.
    with self.nan_guard.watch(cast("mjwarp.Data", self.data)):
      self._batch.step()

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      self._batch.reset()
      return
    ids = np.unique(env_ids.detach().cpu().numpy().astype(np.int64))
    if ids.size:
      self._batch.reset(ids)

  def set_sensor_context(self, ctx: "SensorContext") -> None:
    raise NotImplementedError(
      "Camera and raycast sensors require the Warp backend; use backend='warp'."
    )

  def sense(self) -> None:
    pass
