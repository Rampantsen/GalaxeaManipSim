import time
from contextlib import contextmanager

# import mujoco
# import mujoco.viewer
import numpy as np

# assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."


class PandaPnPEnv:
    def __init__(self, dt=0.002, offscreen_rendering=False):
        # Load the model and data.
        model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
        data = mujoco.MjData(model)

        # It is important to enable gravity compensation, otherwise the control will not work properly.
        # Gravity compensation is enabled in MJCF. The following line seems not to work.
        # model.body_gravcomp[:] = 1.0

        model.opt.timestep = dt
        self._dt = dt

        # Get the dof and actuator ids for the joints we wish to control.
        joint_names = [f"joint{i}" for i in range(1, 8)]
        dof_ids = np.array([model.joint(name).id for name in joint_names])
        actuator_names = [f"actuator{i}" for i in range(1, 8)]
        actuator_ids = np.array([model.actuator(name).id for name in actuator_names])
        gripper_dof_ids = np.array(
            [model.joint(name).id for name in ["finger_joint1", "finger_joint2"]]
        )
        gripper_actuator_ids = np.array(
            [model.actuator(name).id for name in ["actuator8", "actuator9"]]
        )
        obj_id = model.joint("box").id

        self.model = model
        self.data = data
        self.dof_ids = dof_ids
        self.actuator_ids = actuator_ids
        self.gripper_dof_ids = gripper_dof_ids
        self.gripper_actuator_ids = gripper_actuator_ids
        self.obj_id = obj_id

        # Initial poses for the arm, gripper, and object.
        self._init_arm_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
        self._init_gripper_qpos = np.array([0.04, 0.04])
        self._init_obj_qpos = np.array([0.6, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0])
        self._init_target_qpos = np.array([0.5, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0])

        self.viewer = None
        self._elapsed_steps = 0
        self.renderer = None
        self._image_buffer = []
        if offscreen_rendering:
            self.renderer = mujoco.Renderer(model)

    def reset(self, init_obj_qpos=None, init_target_xy=None):
        """Reset the environment to its initial state."""
        self.data.qpos[self.dof_ids] = self._init_arm_qpos
        self.data.qpos[self.gripper_dof_ids] = self._init_gripper_qpos
        self.data.qpos[self.obj_id : self.obj_id + 7] = (
            self._init_obj_qpos if init_obj_qpos is None else init_obj_qpos
        )
        self.data.ctrl[self.actuator_ids] = self._init_arm_qpos
        self.data.ctrl[self.gripper_actuator_ids] = self._init_gripper_qpos

        # Reset velocities and accelerations.
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0

        # Reset the target position.
        if init_target_xy is None:
            init_target_xy = self._init_target_qpos[:2]
        mocap_id = self.model.body("target").mocapid[0]
        self.data.mocap_pos[mocap_id, :2] = init_target_xy

        mujoco.mj_forward(self.model, self.data)

        self._elapsed_steps = 0
        self._image_buffer.clear()

    def step(self, disable_viewer=False, disable_rendering=False):
        if (
            self.renderer is not None
            and self._elapsed_steps % (int(1 / self._dt / 30)) == 0
            and not disable_rendering
        ):
            self.renderer.update_scene(self.data)
            img = self.renderer.render()
            self._image_buffer.append(img)

        mujoco.mj_step(self.model, self.data)
        self._elapsed_steps += 1

        if self.viewer is not None and not disable_viewer:
            self.viewer.sync()  # Sync the viewer with the simulation state
            time.sleep(self._dt)  # Sleep to match the simulation timestep

    def step_for_seconds(self, seconds, disable_viewer=False, disable_rendering=False):
        """Step the environment for a given number of seconds."""
        num_steps = int(seconds / self._dt)
        for _ in range(num_steps):
            self.step(
                disable_viewer=disable_viewer, disable_rendering=disable_rendering
            )

    def create_viewer(self):
        def key_callback(key):
            if key == 61:  # KEY_PLUS
                print("Open gripper")
                self.data.ctrl[self.gripper_actuator_ids] = 0.04
            elif key == 45:  # KEY_MINUS
                print("Close gripper")
                self.data.ctrl[self.gripper_actuator_ids] = 0
            elif key == 82:  # r
                self.reset()
            # else:
            #     print(key)

        viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=True,
            show_right_ui=True,
            key_callback=key_callback,
        )

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)

        # # Enable site frame visualization.
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        return viewer

    @contextmanager
    def launch_viewer(self):
        self.viewer = self.create_viewer()
        try:
            yield self.viewer
        finally:
            self.viewer.close()
            self.viewer = None

    def get_tcp_pose(self):
        site_id = self.model.site("tcp").id
        xpos = self.data.site(site_id).xpos.copy()
        xmat = self.data.site(site_id).xmat.copy()
        return xpos, xmat

    def get_box_pose(self):
        body_id = self.model.body("box").id
        xpos = self.data.body(body_id).xpos.copy()
        xmat = self.data.body(body_id).xmat.copy()
        return xpos, xmat

    def get_target_position(self):
        """Get the current position of the target site."""
        site_id = self.model.site("target").id
        return self.data.site(site_id).xpos.copy()

    def evaluate(self):
        target_xpos = self.get_target_position()
        obj_xpos, _ = self.get_box_pose()
        distance = np.linalg.norm(target_xpos - obj_xpos)
        return distance < 1e-2

    def get_arm_position(self):
        """Get the current joint positions of the arm."""
        return self.data.qpos[self.dof_ids].copy()

    def set_arm_position(self, qpos):
        self.data.ctrl[self.actuator_ids] = qpos

    def get_arm_trajectory(self, target_qpos, expected_time):
        """Generate a trajectory to move the arm to the specified joint positions."""
        current_qpos = self.get_arm_position()
        num_steps = int(np.ceil(expected_time / self._dt))
        trajectory = np.zeros((num_steps, len(self.dof_ids)))

        for i in range(num_steps):
            # Interpolate between current and target joint positions.
            trajectory[i] = (1 - i / num_steps) * current_qpos + (
                i / num_steps
            ) * target_qpos

        return trajectory

    def execute_arm_trajectory(self, trajectory):
        """Execute the given trajectory by setting the joint positions."""
        for qpos in trajectory:
            self.set_arm_position(qpos)
            self.step()

    def open_gripper(self):
        self.data.ctrl[self.gripper_actuator_ids] = [0.04, 0.04]

    def close_gripper(self):
        self.data.ctrl[self.gripper_actuator_ids] = [0, 0]

    def compute_CLIK(
        self,
        target_xpos,
        target_xmat,
        damping=1e-4,
        dt=0.1,
        max_iters=100,
    ):
        model = self.model
        data = self.data
        site_id = model.site("tcp").id

        # Backup the original qpos.
        ori_qpos = data.qpos.copy()

        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, target_xmat)

        # Pre-allocate arrays for results.
        quat = np.zeros(4)
        quat_conj = np.zeros(4)
        error_quat = np.zeros(4)
        jac = np.zeros((6, model.nv))
        diag = damping * np.eye(6)
        # NOTE: It is not an actual twist!
        twist = np.zeros(6)

        for i_iter in range(max_iters):
            # NOTE: Mujoco uses a LOCAL_WORLD_ALIGNED coordinate system
            # https://github.com/google-deepmind/mujoco/issues/1866
            twist[:3] = target_xpos - data.site(site_id).xpos
            mujoco.mju_mat2Quat(quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(quat_conj, quat)
            mujoco.mju_mulQuat(error_quat, target_quat, quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)

            if np.linalg.norm(twist[:3]) < 5e-4 and np.linalg.norm(twist[3:]) < 0.005:
                # print("Converged to target position and orientation.", i_iter)
                break

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Damped least squares.
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)
            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, dt)
            q[self.dof_ids] = np.clip(q[self.dof_ids], *model.jnt_range[self.dof_ids].T)
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
        else:
            print("WARNING: Maximum iterations reached without convergence.")
            print("Last error:", np.linalg.norm(twist[:3]), np.linalg.norm(twist[3:]))

        # Get the joint positions after the last iteration.
        q = data.qpos.copy()

        # Restore the original qpos.
        data.qpos[:] = ori_qpos
        mujoco.mj_forward(model, data)

        return q[self.dof_ids]

    @staticmethod
    def get_grasp_xmat(approaching, closing):
        """Get the grasp frame rotation matrix given approaching and closing vectors.
        Approaching is the direction pointing towards the object. Closing is the direction of the fingers.
        The grasp frame depends on specific robots. For this Panda in Mujoco, approaching is z axis, closing is y axis.
        """
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        xmat = np.stack([ortho, closing, approaching], axis=1)
        return xmat.flatten()

    def export_gif(self, filename="episode.gif"):
        from PIL import Image

        images = [Image.fromarray(img) for img in self._image_buffer]
        if not images:
            return
        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            duration=int(self._dt * 1000),
            loop=0,
        )


def compute_grasp_info_by_obb(
    xpos,
    xmat,
    extents,
    approaching=(0, 0, -1),
    target_closing=None,
    depth=0.0,
    ortho=True,
):
    """Compute grasp info given an oriented bounding box.
    The grasp info includes axes to define grasp frame, namely approaching, closing, orthogonal directions and center.

    Args:
        xpos: position of the object.
        xmat: orientation of the object.
        extents: extents of the object.
        approaching: direction to approach the object
        target_closing: target closing direction, used to select one of multiple solutions
        depth: displacement from hand to tcp along the approaching vector. Usually finger length.
        ortho: whether to orthogonalize closing  w.r.t. approaching.
    """
    # Assume normalized
    approaching = np.array(approaching)
    R = np.array(xmat).reshape(3, 3)

    # Find the axis closest to approaching vector
    angles = approaching @ R  # [3]
    inds0 = np.argsort(np.abs(angles))
    ind0 = inds0[-1]

    # Find the shorter axis as closing vector
    inds1 = np.argsort(extents[inds0[0:-1]])
    ind1 = inds0[0:-1][inds1[0]]
    ind2 = inds0[0:-1][inds1[1]]

    # If sizes are close, choose the one closest to the target closing
    if target_closing is not None and 0.99 < (extents[ind1] / extents[ind2]) < 1.01:
        vec1 = R[:3, ind1]
        vec2 = R[:3, ind2]
        if np.abs(target_closing @ vec1) < np.abs(target_closing @ vec2):
            ind1 = inds0[0:-1][inds1[1]]
            ind2 = inds0[0:-1][inds1[0]]
    closing = R[:3, ind1]

    # Flip if far from target
    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing

    # Reorder extents
    extents = extents[[ind0, ind1, ind2]]

    # Find the origin on the surface
    center = xpos.copy()
    half_size = extents[0] * 0.5
    center = center + approaching * (-half_size + min(depth, half_size))

    def normalize_vector(x, eps=1e-6):
        """normalizes a given numpy array x and if the norm is less than eps, set the norm to 0"""
        x = np.array(x)
        assert x.ndim == 1, x.ndim
        norm = np.linalg.norm(x)
        return np.zeros_like(x) if norm < eps else (x / norm)

    if ortho:
        closing = closing - (approaching @ closing) * approaching
        closing = normalize_vector(closing)

    return approaching, closing, center


def main():
    env = PandaPnPEnv(offscreen_rendering=True)
    rng = np.random.default_rng(42)

    env.viewer = env.create_viewer()
    for episode_id in range(10):
        # Randomly initialize the object position and orientation
        init_xy = rng.uniform([0.5, -0.2], [0.7, 0.2])
        init_quat = rng.uniform(-1, 1, size=4)
        init_quat /= np.linalg.norm(init_quat)  # Normalize the quaternion
        init_obj_qpos = np.hstack([init_xy, 0.04, init_quat])

        # Randomly initialize the target position
        init_target_xy = rng.uniform([0.5, -0.2], [0.7, 0.2])

        env.reset(init_obj_qpos, init_target_xy)
        env.step_for_seconds(0.5, disable_viewer=True, disable_rendering=True)

        # Stage 1: Pre-Grasp (avoid collision with the box)
        box_xpos, box_xmat = env.get_box_pose()
        box_extents = np.array([0.05, 0.05, 0.05])
        approaching, closing, center = compute_grasp_info_by_obb(
            box_xpos,
            box_xmat,
            box_extents,
            approaching=(0, 0, -1),
            target_closing=(1, 0, 0),
            depth=0.025,
        )
        # print("Pregrasp info:", approaching, closing, center)
        grasp_xmat = env.get_grasp_xmat(approaching, closing)
        target_qpos = env.compute_CLIK(
            target_xpos=center + [0, 0, 0.05], target_xmat=grasp_xmat
        )

        trajectory = env.get_arm_trajectory(target_qpos, expected_time=0.5)
        env.execute_arm_trajectory(trajectory)
        env.step_for_seconds(0.1)

        # Check control error
        # print(np.abs(target_qpos - env.get_arm_position()))
        # print(env.get_tcp_pose()[0])

        # Stage 2: Move downwards
        target_qpos = env.compute_CLIK(target_xpos=center, target_xmat=grasp_xmat)
        trajectory = env.get_arm_trajectory(target_qpos, expected_time=0.2)
        env.execute_arm_trajectory(trajectory)
        env.step_for_seconds(0.1)

        # Stage 3: Grasp
        env.close_gripper()
        env.step_for_seconds(0.1)

        # Stage 4: Lift
        target_qpos = env.compute_CLIK(
            target_xpos=center + [0, 0, 0.1], target_xmat=grasp_xmat
        )
        trajectory = env.get_arm_trajectory(target_qpos, expected_time=0.2)
        env.execute_arm_trajectory(trajectory)
        env.step_for_seconds(0.1)

        # Stage 5: Move to target
        target_xpos = env.get_target_position()
        target_qpos = env.compute_CLIK(
            target_xpos=target_xpos + [0, 0, 0.025], target_xmat=grasp_xmat
        )
        trajectory = env.get_arm_trajectory(target_qpos, expected_time=0.5)
        env.execute_arm_trajectory(trajectory)
        env.step_for_seconds(0.2)
        env.open_gripper()
        env.step_for_seconds(0.1)

        print("Success:", env.evaluate())

        # env.export_gif(f"episode_{episode_id}.gif")

    # while viewer.is_running():
    #     env.step()
    env.viewer.close()
    env.viewer = None  # Clean up the viewer if used
    env.renderer = None  # Clean up the renderer if used


if __name__ == "__main__":
    main()
