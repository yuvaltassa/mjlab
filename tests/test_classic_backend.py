"""Tests for the classic (C engine, CPU) simulation backend."""

import mujoco
import numpy as np
import pytest
import torch

from mjlab.sim.classic import ClassicSimulation
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg, make_simulation

PENDULUM_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="cart">
      <joint name="slider" type="slide" axis="1 0 0" range="-3 3"/>
      <geom name="cart_geom" type="box" size=".2 .1 .05" mass="1"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 0.6" size=".045"
          mass=".1"/>
        <site name="tip" pos="0 0 0.6"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="slider" gear="50"/></actuator>
  <sensor>
    <jointpos joint="hinge"/>
    <jointvel joint="hinge"/>
  </sensor>
</mujoco>
"""

NUM_ENVS = 4


def make_classic(num_envs: int = NUM_ENVS, **cfg_kwargs) -> ClassicSimulation:
  model = mujoco.MjModel.from_xml_string(PENDULUM_XML)
  cfg = SimulationCfg(backend="classic", **cfg_kwargs)
  sim = make_simulation(num_envs=num_envs, cfg=cfg, model=model, device="cpu")
  assert isinstance(sim, ClassicSimulation)
  return sim


def test_step_matches_reference() -> None:
  """Stepping through the bridge matches a raw C-engine loop bit-for-bit
  (modulo the float32 presentation cast)."""
  sim = make_classic()
  ctrl = np.linspace(-0.5, 0.5, NUM_ENVS)
  sim.data.ctrl[:, 0] = torch.as_tensor(ctrl, dtype=torch.float32)
  for _ in range(10):
    sim.step()

  ref_model = mujoco.MjModel.from_xml_string(PENDULUM_XML)
  SimulationCfg().mujoco.apply(ref_model)
  for i in range(NUM_ENVS):
    d = mujoco.MjData(ref_model)
    mujoco.mj_forward(ref_model, d)
    d.ctrl[0] = np.float32(ctrl[i])
    for _ in range(10):
      mujoco.mj_step(ref_model, d)
    np.testing.assert_allclose(sim.data.qpos[i].numpy(), d.qpos, rtol=0, atol=1e-6)
    np.testing.assert_allclose(sim.data.qvel[i].numpy(), d.qvel, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
      sim.data.sensordata[i].numpy(), d.sensordata, rtol=0, atol=1e-6
    )


def test_warp_parity_short_horizon() -> None:
  """Classic and Warp backends agree over a short smooth-dynamics horizon."""
  model = mujoco.MjModel.from_xml_string(PENDULUM_XML)
  classic = make_simulation(
    num_envs=NUM_ENVS,
    cfg=SimulationCfg(backend="classic"),
    model=model,
    device="cpu",
  )
  warp_sim = make_simulation(
    num_envs=NUM_ENVS,
    cfg=SimulationCfg(),
    model=mujoco.MjModel.from_xml_string(PENDULUM_XML),
    device="cpu",
  )
  assert isinstance(warp_sim, Simulation)

  ctrl = torch.linspace(-0.3, 0.3, NUM_ENVS).unsqueeze(1)
  classic.data.ctrl[:] = ctrl
  warp_sim.data.ctrl[:] = ctrl
  for _ in range(5):
    classic.step()
    warp_sim.step()
  classic.forward()
  warp_sim.forward()

  for field in ("qpos", "qvel", "xpos", "site_xpos"):
    a = getattr(classic.data, field)
    b = getattr(warp_sim.data, field)
    np.testing.assert_allclose(
      np.asarray(a), np.asarray(b), rtol=1e-3, atol=1e-4, err_msg=field
    )


def test_derived_shapes_and_write_forward() -> None:
  """Matrix fields present as (nworld, n, 3, 3); state writes flow to derived
  quantities after forward()."""
  sim = make_classic()
  assert sim.data.xmat.shape[-2:] == (3, 3)
  assert sim.data.site_xmat.shape == (NUM_ENVS, 1, 3, 3)
  assert sim.data.geom_xmat.shape[-2:] == (3, 3)
  assert sim.data.qpos.dtype == torch.float32

  tip_before = sim.data.site_xpos[:, 0].clone()
  sim.data.qpos[:, 1] = 0.7
  sim.forward()
  tip_after = sim.data.site_xpos[:, 0]
  assert not torch.allclose(tip_before, tip_after)
  # Rotation matrices stay orthonormal after the reshape.
  r = sim.data.xmat[0, 2]
  np.testing.assert_allclose((r @ r.T).numpy(), np.eye(3), atol=1e-6)


def test_expand_randomize_recompute() -> None:
  """Per-env model randomization changes per-env dynamics; mj_setConst-derived
  fields are gathered back."""
  sim = make_classic()
  sim.expand_model_fields(("body_mass", "body_subtreemass"))
  assert sim.model.body_mass.shape[0] == NUM_ENVS

  pole_id = sim.mj_model.body("pole").id
  masses = torch.linspace(0.1, 1.0, NUM_ENVS)
  sim.model.body_mass[:, pole_id] = masses
  sim.recompute_constants(level=None)  # type: ignore[arg-type]
  np.testing.assert_allclose(
    sim.model.body_subtreemass[:, pole_id].numpy(), masses.numpy(), rtol=1e-6
  )

  sim.data.ctrl[:, 0] = 1.0
  sim.step()
  qvel = sim.data.qvel[:, 0]
  # Same force on increasingly heavy systems: strictly decreasing speed.
  assert torch.all(qvel[:-1] > qvel[1:])
  assert torch.all(qvel > 0)


def test_reset_subset_preserves_other_envs() -> None:
  """Resetting a subset restores those envs and leaves other envs' state and
  pending (unflushed) writes untouched."""
  sim = make_classic()
  sim.data.ctrl[:, 0] = 1.0
  for _ in range(5):
    sim.step()
  assert abs(float(sim.data.qpos[1, 0])) > 1e-5

  sim.data.qpos[1, 1] = 0.123  # Pending write for env 1, not yet flushed.
  sim.reset(env_ids=torch.tensor([0]))

  np.testing.assert_allclose(sim.data.qpos[0].numpy(), 0.0, atol=1e-8)
  assert float(sim.data.ctrl[0, 0]) == 0.0  # mj_resetData zeroed env 0's ctrl.
  assert float(sim.data.qpos[1, 1]) == pytest.approx(0.123)
  assert float(sim.data.ctrl[1, 0]) == pytest.approx(1.0)


def test_classic_requires_cpu_and_rejects_variants() -> None:
  model = mujoco.MjModel.from_xml_string(PENDULUM_XML)
  with pytest.raises(ValueError, match="CPU"):
    ClassicSimulation(
      num_envs=1, cfg=SimulationCfg(backend="classic"), model=model, device="cuda:0"
    )


def test_warp_rejects_classic_only_integrator() -> None:
  model = mujoco.MjModel.from_xml_string(PENDULUM_XML)
  cfg = SimulationCfg(mujoco=MujocoCfg(integrator="rk4"))
  with pytest.raises(ValueError, match="classic"):
    Simulation(num_envs=1, cfg=cfg, model=model, device="cpu")


def test_classic_rk4_integrator() -> None:
  sim = make_classic(mujoco=MujocoCfg(integrator="rk4"))
  assert sim.mj_model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
  sim.step()
  assert torch.isfinite(sim.data.qpos).all()


def test_classic_env_smoke() -> None:
  """The full manager stack runs on the classic backend."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg

  cfg = cartpole_balance_env_cfg()
  cfg.scene.num_envs = 8
  cfg.sim.backend = "classic"
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    rew = torch.zeros(env.num_envs)
    obs: dict = {}
    for _ in range(3):
      action = torch.rand((env.num_envs, *env.single_action_space.shape)) - 0.5
      obs, rew, _, _, _ = env.step(action)
    for group in obs.values():
      assert torch.isfinite(torch.as_tensor(group)).all()
    assert torch.isfinite(rew).all()
  finally:
    env.close()


def test_bit_exact_over_chaotic_horizon() -> None:
  """The backend follows a plain mj_step loop bit for bit over a chaotic horizon:
  only written elements reach MjData, so float32 views never round the state."""
  xml = """
  <mujoco>
    <option timestep="0.002"/>
    <worldbody>
      <body pos="0 0 1">
        <joint axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -.5" size=".02" mass="1"/>
        <body pos="0 0 -.5">
          <joint axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -.5" size=".02" mass="1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  """
  q0 = np.array([[2.0 + 0.1 * i, 0.5 + 0.1 * i] for i in range(NUM_ENVS)], np.float32)
  sim = make_simulation(
    num_envs=NUM_ENVS,
    cfg=SimulationCfg(backend="classic"),
    model=mujoco.MjModel.from_xml_string(xml),
    device="cpu",
  )
  sim.data.qpos[:] = torch.as_tensor(q0)

  ref_model = mujoco.MjModel.from_xml_string(xml)
  SimulationCfg().mujoco.apply(ref_model)
  datas = [mujoco.MjData(ref_model) for _ in range(NUM_ENVS)]
  for i, d in enumerate(datas):
    d.qpos[:] = q0[i]
  for _ in range(2000):
    sim.step()
    for d in datas:
      mujoco.mj_step(ref_model, d)
  np.testing.assert_array_equal(
    sim.data.qpos.numpy(), np.stack([d.qpos for d in datas]).astype(np.float32)
  )
