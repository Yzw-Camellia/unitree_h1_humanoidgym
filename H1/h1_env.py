from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import  HumanoidTerrain

class H1(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask
    

    def compute_ref_state(self):
        # Compute the reference state for the robot
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = sin_pos_r * scale_1
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        self.ref_action = 2 * self.ref_dof_pos

    def compute_observations(self):
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,#线速度 3
                                    self.base_ang_vel  * self.obs_scales.ang_vel,#角速度 3
                                    self.projected_gravity,#重力投影 3
                                    self.commands[:, :3] * self.commands_scale,#命令输入 3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,#关节位置 10
                                    self.dof_vel * self.obs_scales.dof_vel,#关节速度 10
                                    self.actions,#动作 10
                                    self.COM#质心坐标 3
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        self.privileged_obs_buf = torch.cat(self.target_positions,
                                            self.check_points,
                                            )
    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:22] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[22:32] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[32:42] = 0. # previous actions
        noise_vec[42:45] = 0.
        if self.cfg.terrain.measure_heights:
            noise_vec[45:166] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec
    def _calculate_foot_positions(self):
        #计算脚的位置
        self.feet_pos = self.rigid_body_state[:, self.feet_indices, :3]
        # print("feet_pos:", self.feet_pos)
        # deta_z = self.feet_pos[:, 0, 2] - (-0.96)
        # print("deta_z:", deta_z)
        self.right_foot_pos = self.feet_pos[:, 0, :]  # 左脚位置 (num_envs, 3)
        self.left_foot_pos = self.feet_pos[:, 1, :]  # 左脚位置 (num_envs, 3)
        
    def _calculate_COM(self):
       self.COM = (self.rigid_body_state[:,:,:3] * self.rigid_body_mass.unsqueeze(1)).sum(dim=1) / self.mass_total
       #print("COM:", self.COM)
    
    def _calculate_target_positions(self):
        """
        Compute the target foothold position for each environment.
        """
        step_width = self.cfg.terrain.step_width# * self.cfg.terrain.horizontal_scale
        step_height = self.cfg.terrain.step_height# * self.cfg.terrain.vertical_scale
        platform_size = self.cfg.terrain.platform_size# * self.cfg.terrain.horizontal_scale
        num_steps = int((self.cfg.terrain.terrain_length - platform_size) / (2 * step_width))
        #print("num_steps:", num_steps)
        y = self.cfg.terrain.terrain_length / 2
        offset = 0.2
        y_right = y - offset
        y_left = y + offset
        #x_start = self.cfg.terrain.terrain_length / 2 + platform_size / 2
        x_values = torch.tensor([8.75 + i * step_width for i in range(num_steps)],dtype=torch.float, device=self.device, requires_grad=False)

        z_values = torch.tensor([num_steps * step_height + i * 0.08 for i in range(num_steps)],dtype=torch.float, device=self.device, requires_grad=False)
                
        # Generate the target position array
        self.target_positions = torch.zeros(self.num_envs, num_steps, 2, 3,dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(num_steps):
            self.target_positions[:, i, 0, :] = torch.tensor([x_values[i], y_left, z_values[i]],dtype=torch.float, device=self.device, requires_grad=False)  # 左腿
            self.target_positions[:, i, 1, :] = torch.tensor([x_values[i], y_right, z_values[i]],dtype=torch.float, device=self.device, requires_grad=False)  # 右腿
        #print("target_position:", self.target_positions)
        
    def _get_env_origins(self):
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            #print("terrain_levels:", self.terrain_levels)
            #print("terrain_types:", self.terrain_types)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
    
    def _draw_CoM_vis(self):
        """
        Plot the projection of the center of mass on the ground
        """
        sphere_COM = gymutil.WireframeSphereGeometry(0.02, 8, 8, None, color=(1,0,1))
        COM_projection = torch.cat((self.COM[:, :2], torch.zeros((self.num_envs,1), device=self.device)), dim=1)
        #print("COM_projection:", COM_projection)
        for i in range(self.num_envs):
            CoM_loc = gymapi.Transform(gymapi.Vec3(*COM_projection[i]), r=None)
            gymutil.draw_lines(sphere_COM, self.gym, self.viewer, self.envs[i], CoM_loc)
    
    def _draw_foot_positions(self):
        sphere_foot = gymutil.WireframeSphereGeometry(0.02, 8, 8, None, color=(1, 1, 1))  # 红色球体
        for i in range(self.num_envs):  
            for foot_id in range(2): 
                foot_pos = self.feet_pos[i, foot_id, :] 
                foot_pos = foot_pos.detach().cpu().numpy().tolist() 
                feet_loc = gymapi.Transform(gymapi.Vec3(*foot_pos), r=None)  
                gymutil.draw_lines(sphere_foot, self.gym, self.viewer, self.envs[i], feet_loc)
    

    
    
    
    def _draw_target_foot_position(self):
        sphere_geometry = gymutil.WireframeSphereGeometry(0.05, 8, 8, None, color=(0, 0, 1))  # 蓝色球体

        for env_id in range(len(self.envs)):
            for step in range(self.target_positions.shape[1]):  # 遍历 num_steps
                for leg in range(2):  # 0: 左腿, 1: 右腿
                    target_position = self.target_positions[env_id, step, leg]  # 取出 (x, y, z)
                    target_loc = gymapi.Transform(gymapi.Vec3(*target_position), r=None)
                    gymutil.draw_lines(sphere_geometry, self.gym, self.viewer, self.envs[env_id], target_loc)
    def _visualization(self):
        self.gym.clear_lines(self.viewer)
        self._draw_heightmap_vis()
        self._draw_CoM_vis()
        self._draw_foot_positions()
        self._draw_target_foot_position()
        #self._draw_point()
#####################################Reward Functions#############################################   
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)    

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6: 8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
    
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
