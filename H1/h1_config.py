from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 166
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 10
        env_spacing = 3. 
        send_timeouts = True 
        episode_length_s = 20
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"
        measured_heights = True
        terrain_length = 15.
        terrain_width = 15
        border_size = 2.5
        step_width = 0.5  
        step_height = -0.08  
        platform_size = 3.0
        num_rows= 1 # number of terrain rows (levels)
        num_cols = 1 # number of terrain cols (types)
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] 
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        terrain_proportions = [0., 0., 0., 1., 0., 0., 0.]
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf'
        name = "h1"
        foot_name = "ankle"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        #这是训练的设置
        class ranges:
            lin_vel_x = [-0., 0.3] # min max [m/s]
            lin_vel_y = [-0., 0.]   # min max [m/s]
            ang_vel_yaw = [0,0]#[-1, 1]    # min max [rad/s]
            heading =[0,0] #[-3.14, 3.14]
    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85
    class init_state(LeggedRobotCfg.init_state):
        pos = [0,0,1.05]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_joint' : -0.2,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_joint' : -0.2,                                         
        }
    class control( LeggedRobotCfg.control ):
        control_type = 'P'
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 150,
                     'knee': 200,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 150,
                     "elbow":100,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'torso': 6,
                     'shoulder': 2,
                     "elbow":2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.5
    class rewards(LeggedRobotCfg.rewards):
        max_dist = 0.50
        
        max_contact_force = 700
        cycle_time = 2.0
        target_jonit_pos_scale = 0.17
        only_positive_rewards = True
        class scales( LeggedRobotCfg.rewards.scales ):
            # reference motion tracking
            joint_pos = 1.6
            feet_clearance = 1.
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.
    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.        
class H1CfgPPo(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.88
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True#裁减值函数损失
        clip_param = 0.2
        entropy_coef = 0.01
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 5500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'H1_ppo'
        run_name = ''
        