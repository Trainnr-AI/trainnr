"""The engine's applier registry: every registered knob restores the
model bit-identically, scales exactly the field its name claims, and an
unknown knob is refused with the list of known ones — the same refusal
the env issues before a trial is spent (envs/robotiq.py)."""

import unittest

from tests._extras import needs_numpy, needs_sim

# A tiny but sufficient model: two damped hinge joints with frictionloss
# (an untouched-field sentinel), position servos with distinct kp, one
# named light, one named camera, two named bodies. Every registered
# applier (physics/variations.py) has something to write to.
ARM = """
<mujoco>
  <worldbody>
    <light name="lamp" pos="0 0 2" diffuse="0.6 0.5 0.4"/>
    <camera name="top" pos="0.1 0.2 0.3"/>
    <body name="upper">
      <joint name="shoulder" type="hinge" axis="0 1 0" damping="0.5"
             frictionloss="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="1"/>
      <body name="lower" pos="0 0 -0.3">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="0.25"
               frictionloss="0.05"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="shoulder" kp="4"/>
    <position joint="elbow" kp="2"/>
  </actuator>
</mujoco>
"""

# One (host, value) per registered knob name. The fixed hosts are the
# vocabulary's words (evaluate/variations.py VariationKeys: joints,
# actuators, lights); mass_scale and offset_m host on a NAMED body and
# camera from ARM. The round-trip test asserts this map covers the
# registry exactly, so a new applier fails here until it is pinned.
DAMPING = 1.7
GAIN = 0.6
DIFFUSE = 1.3
DOUBLE = 2.0  # mass_scale's pin, and the no-compounding factor
OFFSET = (0.01, -0.02, 0.03)

HOSTS_AND_VALUES = {
    "damping_scale": ("joints", DAMPING),
    "gain_scale": ("actuators", GAIN),
    "diffuse_scale": ("lights", DIFFUSE),
    "mass_scale": ("lower", DOUBLE),
    "offset_m": ("top", OFFSET),
}


def _arm():
    import mujoco  # noqa: PLC0415 - sim extra

    return mujoco.MjModel.from_xml_string(ARM)


def _fields(model):
    """Copies of every array any registered applier touches (each
    applier's `snapshot`, physics/variations.py), plus frictionloss as
    the field no applier claims."""
    return {
        "dof_damping": model.dof_damping.copy(),
        "dof_frictionloss": model.dof_frictionloss.copy(),
        "actuator_gainprm": model.actuator_gainprm.copy(),
        "actuator_biasprm": model.actuator_biasprm.copy(),
        "light_diffuse": model.light_diffuse.copy(),
        "body_mass": model.body_mass.copy(),
        "body_inertia": model.body_inertia.copy(),
        "cam_pos": model.cam_pos.copy(),
    }


@needs_sim
class SnapshotRestore(unittest.TestCase):
    def test_every_registered_knob_round_trips_bit_identically(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.physics.variations import (  # noqa: PLC0415
            APPLIERS,
            apply_key,
            restore_all,
            snapshot_all,
        )

        # The pin map and the registry must agree, so a new applier
        # cannot ship without entering this round trip.
        self.assertEqual(set(HOSTS_AND_VALUES), set(APPLIERS))
        model = _arm()
        before = _fields(model)
        nominal = snapshot_all(model)
        self.assertEqual(set(nominal), set(APPLIERS))
        for name, (host, value) in HOSTS_AND_VALUES.items():
            apply_key(model, nominal, f"{host}.{name}", value)
        # Something actually moved before the restore: the round trip
        # must not pass vacuously.
        self.assertFalse(np.array_equal(model.dof_damping, before["dof_damping"]))
        restore_all(model, nominal)
        for field, values in before.items():
            with self.subTest(field=field):
                self.assertTrue(np.array_equal(getattr(model, field), values))


@needs_sim
class Appliers(unittest.TestCase):
    def _applied(self, name: str, value):
        from rq_pipeline.physics.variations import (  # noqa: PLC0415
            apply_key,
            snapshot_all,
        )

        model = _arm()
        before = _fields(model)
        host, _ = HOSTS_AND_VALUES[name]
        apply_key(model, snapshot_all(model), f"{host}.{name}", value)
        return model, before

    def test_damping_scale_multiplies_every_dof_damping(self) -> None:
        import numpy as np  # noqa: PLC0415

        model, before = self._applied("damping_scale", DAMPING)
        # ARM declares damping 0.5 and 0.25; the applier scales the
        # whole vector (physics/variations.py DampingScale.apply).
        self.assertTrue(np.array_equal(before["dof_damping"], [0.5, 0.25]))
        self.assertTrue(
            np.array_equal(model.dof_damping, before["dof_damping"] * DAMPING)
        )
        # frictionloss belongs to no applier and must not move.
        self.assertTrue(
            np.array_equal(model.dof_frictionloss, before["dof_frictionloss"])
        )

    def test_gain_scale_moves_both_kp_terms(self) -> None:
        import numpy as np  # noqa: PLC0415

        model, before = self._applied("gain_scale", GAIN)
        # A position servo compiles kp into gainprm[0] AND -kp into
        # biasprm[1]; scaling only one would move the setpoint instead
        # (GainScale docstring, physics/variations.py).
        self.assertTrue(np.array_equal(before["actuator_gainprm"][:, 0], [4.0, 2.0]))
        self.assertTrue(np.array_equal(before["actuator_biasprm"][:, 1], [-4.0, -2.0]))
        self.assertTrue(
            np.array_equal(
                model.actuator_gainprm[:, 0], before["actuator_gainprm"][:, 0] * GAIN
            )
        )
        self.assertTrue(
            np.array_equal(
                model.actuator_biasprm[:, 1], before["actuator_biasprm"][:, 1] * GAIN
            )
        )

    def test_diffuse_scale_multiplies_every_light(self) -> None:
        import numpy as np  # noqa: PLC0415

        model, before = self._applied("diffuse_scale", DIFFUSE)
        self.assertTrue(
            np.array_equal(model.light_diffuse, before["light_diffuse"] * DIFFUSE)
        )

    def test_mass_scale_touches_only_the_named_body(self) -> None:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        model, before = self._applied("mass_scale", DOUBLE)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lower")
        self.assertEqual(model.body_mass[body], before["body_mass"][body] * DOUBLE)
        # Inertia scales WITH the mass (MassScale docstring), and every
        # other body — world and "upper" — keeps its nominal values.
        self.assertTrue(
            np.array_equal(
                model.body_inertia[body], before["body_inertia"][body] * DOUBLE
            )
        )
        untouched = [i for i in range(model.nbody) if i != body]
        self.assertTrue(
            np.array_equal(model.body_mass[untouched], before["body_mass"][untouched])
        )
        self.assertTrue(
            np.array_equal(
                model.body_inertia[untouched], before["body_inertia"][untouched]
            )
        )

    def test_offset_m_moves_the_named_camera_by_metres(self) -> None:
        import numpy as np  # noqa: PLC0415

        model, before = self._applied("offset_m", OFFSET)
        # An OFFSET, added to the nominal — not an absolute position
        # (CameraOffset.apply, physics/variations.py).
        self.assertTrue(
            np.array_equal(model.cam_pos[0], before["cam_pos"][0] + np.asarray(OFFSET))
        )

    def test_a_camera_offset_needs_three_components(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._applied("offset_m", (0.01, -0.02))
        self.assertIn("three components", str(caught.exception))

    def test_a_knob_on_the_wrong_host_is_refused(self) -> None:
        from rq_pipeline.physics.variations import (  # noqa: PLC0415
            apply_key,
            snapshot_all,
        )

        model = _arm()
        # damping_scale lives on "joints" (_expect_host,
        # physics/variations.py) — a stray host is a spec bug, not a
        # silent no-op.
        with self.assertRaises(ValueError) as caught:
            apply_key(model, snapshot_all(model), "actuators.damping_scale", DAMPING)
        self.assertIn("lives on host", str(caught.exception))

    def test_a_named_host_the_model_lacks_is_refused(self) -> None:
        from rq_pipeline.physics.variations import (  # noqa: PLC0415
            apply_key,
            snapshot_all,
        )

        model = _arm()
        with self.assertRaises(ValueError) as caught:
            apply_key(model, snapshot_all(model), "nobody.mass_scale", DOUBLE)
        self.assertIn("which the model lacks", str(caught.exception))


@needs_sim
class ApplyKey(unittest.TestCase):
    def test_an_unknown_knob_is_refused_naming_the_known_ones(self) -> None:
        from rq_pipeline.physics.variations import (  # noqa: PLC0415
            apply_key,
            snapshot_all,
        )

        # The env applies trial 0's draw through this same apply_key at
        # construction (envs/robotiq.py, "refused before a trial is
        # spent"); this ValueError is that refusal, with the known list
        # built from the registry, never retyped.
        model = _arm()
        with self.assertRaises(ValueError) as caught:
            apply_key(model, snapshot_all(model), "joints.bogus_scale", 1.0)
        message = str(caught.exception)
        self.assertIn("unknown variation", message)
        self.assertIn("'joints.bogus_scale'", message)
        for name in HOSTS_AND_VALUES:
            self.assertIn(name, message)

    def test_the_refusal_leaves_the_model_untouched(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.physics.variations import (  # noqa: PLC0415
            apply_key,
            snapshot_all,
        )

        model = _arm()
        before = _fields(model)
        with self.assertRaises(ValueError):
            apply_key(model, snapshot_all(model), "joints.bogus_scale", 1.0)
        for field, values in before.items():
            self.assertTrue(np.array_equal(getattr(model, field), values))

    def test_applying_from_nominal_never_compounds(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.physics.variations import (  # noqa: PLC0415
            apply_key,
            snapshot_all,
        )

        # apply always works FROM the snapshot, so two applies of x2
        # give x2, not x4 — the "draws never compound across resets"
        # promise (Variation docstring, evaluate/variations.py).
        model = _arm()
        before = model.dof_damping.copy()
        nominal = snapshot_all(model)
        apply_key(model, nominal, "joints.damping_scale", DOUBLE)
        apply_key(model, nominal, "joints.damping_scale", DOUBLE)
        self.assertTrue(np.array_equal(model.dof_damping, before * DOUBLE))


@needs_numpy
class Registry(unittest.TestCase):
    def test_registering_a_name_twice_is_refused(self) -> None:
        from rq_pipeline.physics.variations import APPLIERS, applier  # noqa: PLC0415

        class Impostor:
            def snapshot(self, model):
                return None

            def restore(self, model, nominal) -> None:
                pass

            def apply(self, model, nominal, host, value) -> None:
                pass

        # The duplicate check fires BEFORE registration (applier(),
        # physics/variations.py), so the registry is left unpolluted.
        census = dict(APPLIERS)
        with self.assertRaises(ValueError) as caught:
            applier("damping_scale")(Impostor)
        self.assertIn("registered twice", str(caught.exception))
        self.assertEqual(dict(APPLIERS), census)

    def test_known_keys_lists_the_registry_sorted(self) -> None:
        from rq_pipeline.physics.variations import known_keys  # noqa: PLC0415

        names = known_keys().split(", ")
        self.assertEqual(names, sorted(names))
        for name in HOSTS_AND_VALUES:
            self.assertIn(name, names)

    def test_every_registered_applier_satisfies_the_protocol(self) -> None:
        from rq_pipeline.physics.variations import APPLIERS, Applier  # noqa: PLC0415

        for name, registered in APPLIERS.items():
            with self.subTest(name=name):
                self.assertIsInstance(registered, Applier)


if __name__ == "__main__":
    unittest.main()
