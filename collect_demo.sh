python -m galaxea_sim.scripts.collect_demos --env-name R1ProBlocksStackEasy-traj_aug  --num-demos 2000   --feature normal
python -m galaxea_sim.scripts.replay_demos --env-name R1ProBlocksStackEasy-traj_aug --num-demos 1000   --feature normal
