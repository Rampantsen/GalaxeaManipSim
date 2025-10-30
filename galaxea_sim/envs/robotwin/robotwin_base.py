import sapien
import numpy as np

from typing import Literal

from galaxea_sim.envs.base.bimanual_manipulation import BimanualManipulationEnv
from galaxea_sim.utils.robotwin_utils import create_table, create_box

class RoboTwinBaseEnv(BimanualManipulationEnv):
    def __init__(
        self,
        robot_class,
        robot_kwargs: dict = {},
        controller_type: str = "bimanual_joint_position",
        control_freq: int = 15,
        timestep: float = 0.01,
        headless: bool = True,
        obs_mode: Literal["state", "image"] = "image",  
        variant_idx: int = 0,
        ray_tracing: bool = False,
        table_type: str = None,
    ):
        self.variant_idx = variant_idx
        self.table_type = table_type
        super().__init__(
            robot_class,
            robot_kwargs,
            controller_type,
            control_freq,
            timestep,
            headless,
            obs_mode,
            ray_tracing
        )
    
    def _build_world(self):
        super()._build_world()
        self._setup_table(self.table_type)
        self._setup_wall()
        self.reset_world()

    @property
    def table_length(self):
        return 1.2
    
    @property
    def table_width(self):
        return 0.7
    
    @property
    def table_height(self):
        if self.robot_name == "r1_lite":
            return 0.7
        else:
            return 0.750
        
    @property
    def tabletop_center_x(self):
        if self.robot_name == "r1_lite":
            return 0.5
        else:
            return 0.7

    def _setup_table(self,table_type):
        self.table_static = True
        self.tabletop_center_in_world = np.array([self.tabletop_center_x, 0, self.table_height])
        
        # 如果 table_type 为 None，使用默认纯色桌子
        if table_type is None or table_type == "plain":
            self.table = create_table(
                self._scene,
                sapien.Pose(p=self.tabletop_center_in_world),
                length=self.table_length,
                width=self.table_width,
                height=self.table_height,
                thickness=0.05,
                is_static=self.table_static,
            )
        elif table_type == "red":
            # 使用 Wood085A PBR 材质（完整的纹理贴图）
            texture_base_path = "/home/sen/workspace/galaxea/GalaxeaManisim/galaxea_sim/assets/robotwin_models/table/Wood069_2K-PNG"
            
            self.table = create_table(
                self._scene,
                sapien.Pose(p=self.tabletop_center_in_world),
                length=self.table_length,
                width=self.table_width,
                height=self.table_height,
                thickness=0.05,
                is_static=self.table_static,
                texture_path=f"{texture_base_path}/Wood069_2K-PNG_Color_rotated.png",
                roughness_path=f"{texture_base_path}/Wood069_2K-PNG_Roughness_rotated.png",
                normal_path=f"{texture_base_path}/Wood069_2K-PNG_NormalGL_rotated.png",
            )

        elif table_type == "white":
            texture_base_path = "/home/sen/workspace/galaxea/GalaxeaManisim/galaxea_sim/assets/robotwin_models/table/Poliigon_WoodVeneerOak_7760"
            self.table = create_table(
                self._scene,
                sapien.Pose(p=self.tabletop_center_in_world),
                length=self.table_length,
                width=self.table_width,
                height=self.table_height,
                thickness=0.05,
                is_static=self.table_static,
                texture_path=f"{texture_base_path}/Poliigon_WoodVeneerOak_7760_BaseColor.jpg",
                roughness_path=f"{texture_base_path}/Poliigon_WoodVeneerOak_7760_Roughness.jpg",
                normal_path=f"{texture_base_path}/Poliigon_WoodVeneerOak_7760_Normal.png",
                brightness=1.5,  # 增加到2.5，更白更亮
            )
        # elif table_type == "Wood068":
        #     texture_base_path = "/home/sen/workspace/galaxea/GalaxeaManisim/galaxea_sim/assets/robotwin_models/table/Wood068_4K-PNG"
        #     self.table = create_table(
        #         self._scene,
        #         sapien.Pose(p=self.tabletop_center_in_world),
        #         length=self.table_length,
        #         width=self.table_width,
        #         height=self.table_height,
        #         thickness=0.05,
        #         is_static=self.table_static,
        #         texture_path=f"{texture_base_path}/Wood068_4K-PNG_Color_rotated.png",
        #         roughness_path=f"{texture_base_path}/Wood068_4K-PNG_Roughness_rotated.png",
        #         normal_path=f"{texture_base_path}/Wood068_4K-PNG_NormalGL_rotated.png",
        #         brightness=2,  # 增加到2.5，更白更亮
        #     )
    def _setup_wall(self):
        self.wall = create_box(
            self._scene,
            sapien.Pose(p=[1.0 + self.tabletop_center_in_world[0], 0, 1.5]),
            half_size=[0.6, 3, 1.5],
            color=(1, 0.9, 0.9), 
            name='wall',
        )
        
    def _add_light(self):
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        shadow = True
        direction_lights = [[[0.5 + 0.7, 0, -1], [0.5, 0.5, 0.5]]]
        for direction_light in direction_lights:
            self._scene.add_directional_light(
                direction_light[0], direction_light[1], shadow=shadow
            )
        point_lights = [[[0.7, 1, 1.8], [1, 1, 1]], [[0.7, -1, 1.8], [1, 1, 1]]]
        for point_light in point_lights:
            self._scene.add_point_light(point_light[0], point_light[1], shadow=shadow)