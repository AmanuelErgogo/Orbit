"""


"""
from omni.isaac.core.prims import XFormPrimView   # GeometryPrim, RigidPrim, RigidPrimView, GeometryPrimView, 
from omni.isaac.core.utils.stage import get_current_stage
import numpy as np

import math
from pxr import UsdLux, UsdGeom, Sdf, Gf, Vt, UsdPhysics, PhysxSchema
from omni.physx.scripts import utils
from omni.physx.scripts import physicsUtils, particleUtils
import omni.physx.bindings._physx as physx_settings_bindings
import omni.timeline


class DeformableObject():
    """deformable object class
    Ref: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_deformable_body_a_p_i.html#a8f60f84ad6738d7e37f798f35d6e060e
    """
    def __init__(self, deform_path: str) -> None:
        self.prim_path = deform_path

    def get_deformable_body_from_stage(self):
        """ Returns PhysxDeformableBodyAPI to access local properties of deformable body """
        prim = get_current_stage().GetPrimAtPath(self.prim_path)
        return PhysxSchema.PhysxDeformableBodyAPI(prim)

    def get_object_view(self):
        """ Return XformView to access global properties of deformable object"""
        return XFormPrimView(prim_paths_expr=self.prim_path, name="DeformableSphere")
    
    def get_object_pose_w(self, deformable_object):
        return deformable_object.get_world_poses()
    
    def get_collision_indeces(self, deformable_body):
        return deformable_body.GetCollisionIndicesAttr().Get()
    
    def get_simulation_points(self, deformable_body):
        return deformable_body.GetSimulationPointsAttr().Get()
    
    """ Ref --- how to get global prop of soft body
    

    deform_path = “/World/Cube”
    prim = get_current_stage().GetPrimAtPath(deform_path)
    deformable_local = PhysxSchema.PhysxDeformableBodyAPI(prim)
    deformable_global = XFormPrimView(prim_paths_expr=deform_path, name=“Deformable_Cube”)

    local_collision_point = np.array(deformable_local.GetCollisionPointsAttr().Get())
    for i in range(3):
    local_collision_point[:,i] = local_collision_point[:,i]

    global_collision_point = deformable_global.get_world_poses()[0]
    final = (local_collision_point + global_collision_point) # N X 3 Numpy Array

    """


class FoodParticle():
    def __init__(self):
        self._time = 0
        self._is_running = False
        self._rng_seed = 42
        self._rng = np.random.default_rng(self._rng_seed)
        self._ball_spawn_interval = 0.4
        self._next_ball_time = self._ball_spawn_interval
        self._num_balls = 0
        self._num_balls_to_spawn = 25
        self._num_colors = 20
  
    def particle_sphere(self, radius, particleSpacing):
        points = []
        dim = math.ceil(2 * radius / particleSpacing)
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    x = i * particleSpacing - radius + self._rng.uniform(-0.05, 0.05)
                    y = j * particleSpacing - radius + self._rng.uniform(-0.05, 0.05)
                    z = k * particleSpacing - radius + self._rng.uniform(-0.05, 0.05)
                    d2 = x * x + y * y + z * z
                    if d2 < radius * radius:
                        points.append(Gf.Vec3f(x, y, z))
        return points
    
    def create_colors(self):

        fractions = np.linspace(0.0, 1.0, self._num_colors)
        colors = []

        for frac in fractions:
            colors.append(self.create_color(frac))

        return colors

    def create_color(self, frac):

        # HSL to RGB conversion
        hue = frac
        saturation = 1.0
        luminosity = 0.5

        hue6 = hue * 6.0
        modulo = Gf.Vec3f((hue6 + 0.0) % 6.0, (hue6 + 4.0) % 6.0, (hue6 + 2.0) % 6.0)
        absolute = Gf.Vec3f(abs(modulo[0] - 3.0), abs(modulo[1] - 3.0), abs(modulo[2] - 3.0))
        rgb = Gf.Vec3f(
            Gf.Clampf(absolute[0] - 1.0, 0.0, 1.0),
            Gf.Clampf(absolute[1] - 1.0, 0.0, 1.0),
            Gf.Clampf(absolute[2] - 1.0, 0.0, 1.0),
        )

        linter = Gf.Vec3f(1.0) * (1.0 - saturation) + rgb * saturation
        rgb = luminosity * linter

        return rgb

    def create_ball(self, stage, pos):

        basePos = Gf.Vec3f(11.0, 12.0, 7.0) + pos

        positions_list = [x + basePos for x in self.ball]
        velocities_list = [Gf.Vec3f(10, 10, 0.0)] * len(positions_list)
        color = Vt.Vec3fArray([self.colors[self._num_balls % self._num_colors]])

        particlePointsPath = Sdf.Path("/particles" + str(self._num_balls))

        if self.usePointInstancer:
            particlePrim = particleUtils.add_physx_particleset_pointinstancer(
                stage,
                particlePointsPath,
                positions_list,
                velocities_list,
                self.particleSystemPath,
                self_collision=True,
                fluid=True,
                particle_group=0,
                particle_mass=0.001,
                density=0.0,
            )

            prototypeStr = str(particlePointsPath) + "/particlePrototype0"
            gprim = UsdGeom.Sphere.Define(stage, Sdf.Path(prototypeStr))
            gprim.CreateDisplayColorAttr(color)
            gprim.CreateRadiusAttr().Set(self.fluid_rest_offset)
        else:
            particlePrim = particleUtils.add_physx_particleset_points(
                stage,
                particlePointsPath,
                positions_list,
                velocities_list,
                [2*self.fluid_rest_offset]*len(positions_list),
                self.particleSystemPath,
                self_collision=True,
                fluid=True,
                particle_group=0,
                particle_mass=0.001,
                density=0.0
            )

            particlePrim.CreateDisplayColorAttr(color)

        self.particlePrims.append(particlePrim)