import gymnasium as gym
from galaxea_sim.robots.r1 import R1Robot
from galaxea_sim.robots.r1_pro import R1ProRobot
from galaxea_sim.robots.r1_lite import R1LiteRobot

R1_INIT_QPOS = [
    0.70050001,
    -1.40279996,
    -0.99959999,
    0.0,
    0,
    0,
    1.57,
    1.57,
    -0.96,
    -0.96,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

R1PRO_INIT_QPOS = [
    0.7,
    -1.4,
    -0.9,
    0.0,
    -0.4,
    -0.4,
    1.3,
    -1.3,
    -0.7,
    0.7,
    -1.57,
    -1.57,
    1.3,
    -1.3,
    -0.4,
    -0.4,
    -0.8,
    0.8,
    0,
    0,
    0,
    0,
]

R1LITE_INIT_QPOS = [
    -0.77,
    1.57,
    0.8,
    0,
    0,
    1.57,
    1.57,
    -0.96,
    -0.96,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

gym.register(
    id="R1DualBottlesPickEasy-v0",
    entry_point="galaxea_sim.envs.robotwin.dual_bottles_pick_easy:DualBottlesPickEasyEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=200,
)

gym.register(
    id="R1ProDualBottlesPickEasy-v0",
    entry_point="galaxea_sim.envs.robotwin.dual_bottles_pick_easy:DualBottlesPickEasyEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1ProRobot,
        robot_kwargs=dict(init_qpos=R1PRO_INIT_QPOS),
        headless=False,
    ),
    max_episode_steps=200,
)

gym.register(
    id="R1ProDualBottlesPickEasy-traj_aug",
    entry_point="galaxea_sim.envs.robotwin.dual_bottles_pick_easy-traj_aug:DualBottlesPickEasyEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1ProRobot,
        robot_kwargs=dict(init_qpos=R1PRO_INIT_QPOS),
        headless=False,
    ),
    max_episode_steps=200,
)


gym.register(
    id="R1BlockHammerBeat-v0",
    entry_point="galaxea_sim.envs.robotwin.block_hammer_beat:BlockHammerBeatEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=200,
)

gym.register(
    id="R1BlockHandover-v0",
    entry_point="galaxea_sim.envs.robotwin.block_handover:BlockHandoverEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=400,
)


gym.register(
    id="R1BlocksStackEasy-v0",
    entry_point="galaxea_sim.envs.robotwin.blocks_stack_easy:BlocksStackEasyEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot,
        robot_kwargs=dict(init_qpos=R1_INIT_QPOS),
        headless=False,
        ray_tracing=False,
    ),
    max_episode_steps=350,
)

gym.register(
    id="R1ProBlocksStackEasy-v0",
    entry_point="galaxea_sim.envs.robotwin.blocks_stack_easy:BlocksStackEasyEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1ProRobot,
        robot_kwargs=dict(init_qpos=R1PRO_INIT_QPOS),
        headless=False,
        ray_tracing=False,
    ),
    max_episode_steps=350,
)
gym.register(
    id="R1ProBlocksStackEasy-traj_aug",
    entry_point="galaxea_sim.envs.robotwin.blocks_stack_easy_traj_aug:BlocksStackEasyTrajAugEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1ProRobot,
        robot_kwargs=dict(init_qpos=R1PRO_INIT_QPOS),
        headless=False,
        ray_tracing=False,
    ),
    max_episode_steps=350,
)

gym.register(
    id="R1ProBlocksStackHard-v0",
    entry_point="galaxea_sim.envs.robotwin.blocks_stack_hard:BlocksStackHardEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1ProRobot,
        robot_kwargs=dict(init_qpos=R1PRO_INIT_QPOS),
        headless=False,
    ),
    max_episode_steps=500,
)
gym.register(
    id="R1ProBlocksStackHard-traj_aug",
    entry_point="galaxea_sim.envs.robotwin.blocks_stack_hard_traj_aug:BlocksStackHardEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1ProRobot,
        robot_kwargs=dict(init_qpos=R1PRO_INIT_QPOS),
        headless=False,
    ),
    max_episode_steps=500,
)

gym.register(
    id="R1ProDualBottlesPickHard-v0",
    entry_point="galaxea_sim.envs.robotwin.dual_bottles_pick_hard:DualBottlesPickHardEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1ProRobot,
        robot_kwargs=dict(init_qpos=R1PRO_INIT_QPOS),
        headless=False,
    ),
    max_episode_steps=200,
)

gym.register(
    id="R1ContainerPlace-v0",
    entry_point="galaxea_sim.envs.robotwin.container_place:ContainerPlaceEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=200,
)


gym.register(
    id="R1DiverseBottlesPick-v0",
    entry_point="galaxea_sim.envs.robotwin.diverse_bottles_pick:DiverseBottlesPickEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=200,
)

gym.register(
    id="R1DualBottlesPickHard-v0",
    entry_point="galaxea_sim.envs.robotwin.dual_bottles_pick_hard:DualBottlesPickHardEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=200,
)

gym.register(
    id="R1DualShoesPlace-v0",
    entry_point="galaxea_sim.envs.robotwin.dual_shoes_place:DualShoesPlaceEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=200,
)


gym.register(
    id="R1EmptyCupPlace-v0",
    entry_point="galaxea_sim.envs.robotwin.empty_cup_place:EmptyCupPlaceEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=400,
)


gym.register(
    id="R1MugHangingEasy-v0",
    entry_point="galaxea_sim.envs.robotwin.mug_hanging_easy:MugHangingEasyEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=450,
)


gym.register(
    id="R1MugHangingHard-v0",
    entry_point="galaxea_sim.envs.robotwin.mug_hanging_hard:MugHangingHardEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=450,
)

gym.register(
    id="R1LiteMugHangingHard-v0",
    entry_point="galaxea_sim.envs.robotwin.mug_hanging_hard:MugHangingHardEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1LiteRobot,
        robot_kwargs=dict(init_qpos=R1LITE_INIT_QPOS),
        headless=False,
    ),
    max_episode_steps=450,
)

gym.register(
    id="R1PickAppleMessy-v0",
    entry_point="galaxea_sim.envs.robotwin.pick_apple_messy:PickAppleMessyEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=200,
)

gym.register(
    id="R1PutAppleCabinet-v0",
    entry_point="galaxea_sim.envs.robotwin.put_apple_cabinet:PutAppleCabinetEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=400,
)


gym.register(
    id="R1ShoePlace-v0",
    entry_point="galaxea_sim.envs.robotwin.shoe_place:ShoePlaceEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=300,
)

gym.register(
    id="R1ToolAdjust-v0",
    entry_point="galaxea_sim.envs.robotwin.tool_adjust:ToolAdjustEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=150,
)


gym.register(
    id="R1HammerBlock-v0",
    entry_point="galaxea_sim.envs.robotwin.hammer_block:HammerBlockEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, robot_kwargs=dict(init_qpos=R1_INIT_QPOS), headless=False
    ),
    max_episode_steps=350,
)

gym.register(
    id="R1BottleCup-v0",
    entry_point="galaxea_sim.envs.robotwin.bottle_cup:BottleCupEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot,
        robot_kwargs=dict(init_qpos=R1_INIT_QPOS),
        headless=False,
    ),
    max_episode_steps=400,
)
