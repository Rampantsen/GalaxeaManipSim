python -m galaxea_sim.scripts.collect_demos --env-name R1ProBlocksStackEasy-traj_aug  --num-demos 1500   --feature no-retry &
wait
python -m galaxea_sim.scripts.replay_demos --env-name R1ProBlocksStackEasy-traj_aug --num-demos 1000   --feature no-retry
