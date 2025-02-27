# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Domain randomization for the Go1 environment."""

import jax
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1

FLOOR_FRICTION_MIN = 0.4
FLOOR_FRICTION_MAX = 1.0

DOF_FRICTIONLOSS_SCALE_MIN = 0.9
DOF_FRICTIONLOSS_SCALE_MAX = 1.1

ARMAUTRE_SCALE_MIN = 1.0
ARMAUTRE_SCALE_MAX = 1.05

COM_POS_JITTER_MIN = -0.05
COM_POS_JITTER_MAX = 0.05

LINK_MASS_SCALE_MIN = 0.9
LINK_MASS_SCALE_MAX = 1.1

ADDED_MASS_MIN = -0.5
ADDED_MASS_MAX = 0.5

INIT_QPOS_JITTER_MIN = -0.05
INIT_QPOS_JITTER_MAX = 0.05


def domain_randomize(model: mjx.Model, rng: jax.Array):
    @jax.vmap
    def rand_dynamics(rng):
        # Floor friction: =U(0.4, 1.0).
        rng, key = jax.random.split(rng)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
            jax.random.uniform(
                key, minval=FLOOR_FRICTION_MIN, maxval=FLOOR_FRICTION_MAX
            )
        )

        # Scale static friction: *U(0.9, DOF_FRICTIONLOSS_SCALE_MAX).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
            key,
            shape=(12,),
            minval=DOF_FRICTIONLOSS_SCALE_MIN,
            maxval=DOF_FRICTIONLOSS_SCALE_MAX,
        )
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        # Scale armature: *U(ARMAUTRE_SCALE_MIN, ARMAUTRE_SCALE_MAX).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[6:] * jax.random.uniform(
            key, shape=(12,), minval=ARMAUTRE_SCALE_MIN, maxval=ARMAUTRE_SCALE_MAX
        )
        dof_armature = model.dof_armature.at[6:].set(armature)

        # Jitter center of mass positiion: +U(COM_POS_JITTER_MIN, COM_POS_JITTER_MAX).
        rng, key = jax.random.split(rng)
        dpos = jax.random.uniform(
            key, (3,), minval=COM_POS_JITTER_MIN, maxval=COM_POS_JITTER_MAX
        )
        body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
            model.body_ipos[TORSO_BODY_ID] + dpos
        )

        # Scale all link masses: *U(LINK_MASS_SCALE_MIN, LINK_MASS_SCALE_MAX).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key,
            shape=(model.nbody,),
            minval=LINK_MASS_SCALE_MIN,
            maxval=LINK_MASS_SCALE_MAX,
        )
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        # Add mass to torso: +U(ADDED_MASS_MIN, ADDED_MASS_MAX).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=ADDED_MASS_MIN, maxval=ADDED_MASS_MAX)
        body_mass = body_mass.at[TORSO_BODY_ID].set(body_mass[TORSO_BODY_ID] + dmass)

        # Jitter qpos0: +U(INIT_QPOS_JITTER_MIN, INIT_QPOS_JITTER_MAX).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[7:].set(
            qpos0[7:]
            + jax.random.uniform(
                key,
                shape=(12,),
                minval=INIT_QPOS_JITTER_MIN,
                maxval=INIT_QPOS_JITTER_MAX,
            )
        )

        return (
            geom_friction,
            body_ipos,
            body_mass,
            qpos0,
            dof_frictionloss,
            dof_armature,
        )

    (
        friction,
        body_ipos,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "body_ipos": 0,
            "body_mass": 0,
            "qpos0": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
        }
    )

    model = model.tree_replace(
        {
            "geom_friction": friction,
            "body_ipos": body_ipos,
            "body_mass": body_mass,
            "qpos0": qpos0,
            "dof_frictionloss": dof_frictionloss,
            "dof_armature": dof_armature,
        }
    )

    return model, in_axes
