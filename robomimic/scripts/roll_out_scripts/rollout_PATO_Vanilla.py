"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
import pdb
import matplotlib.pyplot as plt
import torch
import wandb
from wandb import plot
import cv2
from robomimic.scripts.gaze_sensor.gaze_socket_client import SimpleClient
from robomimic.utils.oculus_controller import VRPolicy
import robomimic.utils.transform_utils as T

import robomimic
import robomimic.utils.gaze_data_utils as GazeUtils
import robomimic.utils.camera_utils as CameraUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy, RolloutPolicy_HBC #TODO: one of the main changes
from screeninfo import get_monitors
from robomimic.scripts.roll_out_scripts.rollout_utils import * 


def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, rollout_number=None, human_expert=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy_HBC)
    # assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    video_count = 0  # video frame counter
    total_reward = 0.
    # subgoal_poll_count = 0
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    subgoal_stats = dict( var_eef_world_coordinates=[], var_eef_pixel_locations=[]) # We are directly storing the variances instead of the actual input values

    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))

    try:
        for step_i in range(0,horizon): #TODO: Check the indexing logic


            interval = 50 #NOTE(aliang) : you have to manually change the subgoal update interval
            #TODO: Currently manually inputting the subgoal update interval, improve this

            # Check if time to update subgoal
            if step_i % interval == 0:

                # get subgoal samples from planner (VAE)
                sg_proposals = policy.subgoal_proposals(ob=obs)

                print("The shape of the subgoal proposal is", sg_proposals['robot0_eef_pos'].shape) # Debugging
                world_points = subgoal_ee_pos(sg_proposals) # Extracting the eef position of all the subgoal samples, EEF world frame coordinates

                pp_sgs = pp_sgs[:, ::-1] #NOTE(dhanush): This reversing is done as the format is reverted

                var_world_sg, var_pixel_sg = calculate_max_column_variance(world_points), calculate_max_column_variance(pp_sgs) #Computing Column wise variance and taking the max
                #TODO: Do not only track the maximum, track all the (x,y,z) - World points, (x,y) - Gaze Points

                # To keep track of the variance
                subgoal_stats['var_eef_world_coordinates'].append(var_world_sg) # Adding to dict
                subgoal_stats['var_eef_pixel_locations'].append(var_pixel_sg)   # Adding to dict

                subgoal_index = np.random.randint(0, sg_proposals['robot0_eef_pos'].shape[1]) # NOTE(dhanush) : CHOOSE ONE OF THE SAMPLES FROM THE PLANNER
                choosen_subgoal = choose_subgoal(sg_proposals, subgoal_index) # Choosing by index from the samples

                policy.set_subgoal(choosen_subgoal) # Setting the subgoal using setter function


            # print("Subgoal Being used is", choosen_subgoal['robot0_gripper_qpos'])
            # NOTE(dhanush) : THIS THRESHOLD CORRESPONDS TO TASK UNCERTAINTY 
            TU_threshold = 0.0 

            use_human_expert = (var_world_sg > TU_threshold) # NOTE (dhanush) : THIS BOOLEAN IS USED TO CALL HUMAN EXPERT

            if use_human_expert:
                
                while True: # TODO(dhanush) : VERIFY THIS LOGIC OF HUMAN EXPERT

                    # HACK(dhanush) : VR Controller Observation
                    print("I NEED HELP FROM HUMAN, PROVIDE ACTION !")
                    VR_obs_dict = {}
                    VR_obs_dict["robot_state"] = {}
                    
                    # NOTE(dhanush) : FIXING THE FORMAT 
                    ee_pos = env.robots[0].controller.ee_pos
                    ee_ori_mat = env.robots[0].controller.ee_ori_mat
                    ee_euler = T.mat2euler(ee_ori_mat)

                    # NOTE(dhanush) : FEEDING THE VR CONTROLLER THE OBS_DICT
                    VR_obs_dict['robot_state']['cartesian_position'] = np.concatenate([ee_pos, ee_euler]) # 6 coords. 3 xyz ee pose, 3 euler angles
                    VR_obs_dict['robot_state']['gripper_position'] = 0
                     

                    if  not policy.__dict__['_state']['movement_enabled']: # Step only when enabled
                        act = policy.forward(VR_obs_dict) # GIVING OBSERVATION TO VR POLICY
                        break

            else:
                act = policy(ob=obs, goal=None)

            # play action
            next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0], width=3440, height=1440)  #TODO: need to put the correct numbers, being overridden by OpenCv ?
                
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        current_frame = env.render(mode="dual", height=1440, width=3440, camera_name=cam_name) #TODO: check dual render in env_robosuite.py


                        edited_frame = cv2.drawMarker(np.uint8(current_frame.copy()), (int(pp_sgs[subgoal_index][0]), 
                                                                                  int(pp_sgs[subgoal_index][1])), color=(0, 255, 0), 
                                                                                  markerType=cv2.MARKER_CROSS, markerSize=50, 
                                                                                  thickness=2) # TODO: verify if this is correct maker 
                        # TODO: check if np.unit8 
                        if step_i % interval ==0:
                            for point in pp_sgs:
                                x, y = int(point[0]), int(point[1])
                                # y, x = int(point[0]), int(point[1]) #NOTE(dhanush): gaze in the pixel converted subgoals have reverted order. 
                                # Draw a marker for each point
                                edited_frame = cv2.drawMarker(np.uint8(edited_frame.copy()), (x, y), 
                                                color=(255, 0, 0),  # Different color for these markers
                                                markerType=cv2.MARKER_CROSS, 
                                                markerSize=10,  # Adjust size as needed
                                                thickness=1)    # Adjust thickness as needed
                        

                        texts = [
                            f"Sim step: {step_i}",
                            f"Current subgoal: {pp_sgs[subgoal_index]}",
                            f"Task Uncertainty world: {var_world_sg}",
                            f"Task Uncertainty pixels: {var_pixel_sg}"
                        ]

                        font_scale = 1.0
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        thickness = 1
                        text_color = (0, 0, 255)

                        # Calculate the maximum width of text and height
                        text_size = [cv2.getTextSize(text, font, font_scale, thickness)[0] for text in texts]
                        max_text_width = max([size[0] for size in text_size])
                        max_text_height = max([size[1] for size in text_size])

                        # Starting Y position (bottom of the frame)
                        start_y = edited_frame.shape[0] - len(texts) * max_text_height

                        for i, text in enumerate(texts):
                            position = (edited_frame.shape[1] - max_text_width - 10, start_y + i * max_text_height)  # 10 pixels padding
                            cv2.putText(edited_frame, text, position, font, font_scale, text_color, thickness)

                        video_img.append(edited_frame)
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])


    ### We also return the subgoal 3D world co-ordinates and the pixel location corresponding to them


    print("ROLLOUT DONE!") # NOTE(dhanush) : Useful when not rendering in the dual mode
    return stats, traj, subgoal_stats


def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    # assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True, HBC=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=(args.video_path is not None), 
        verbose=True,
    )

    # env, _ = FileUtils.env_from_checkpoint(
    #     ckpt_dict=ckpt_dict, 
    #     env_name=args.env, 
    #     render=True, 
    #     render_offscreen=True, 
    #     verbose=True,
    # )


    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    # Setup wandb Script
    # wandb_run = initialize_wandb("my_rollout_project", "multiple_rollout_experiment")
        
    #VR Controller Code here
    VR_human_expert = VRPolicy(
        max_lin_vel=1.0,
        max_rot_vel=0.5,
        pos_action_gain=2.0,
        rot_action_gain=0.5,
        rmat_reorder=[3, 1, 2,4]
    )

 
    rollout_stats = []
    subgoal_stats_cumulative = []
    for i in range(rollout_num_episodes):
        stats, traj, subgoal_stats = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names, rollout_number=i,
            human_expert= VR_human_expert
        )


        rollout_stats.append(stats)

        # TODO (dhanush) : Fix the stats tracking stuff
        # variance_world_coordiantes = calculate_max_column_variance(subgoal_stats['eef_world_coordinates'])
        # variance_pixel_locations = calculate_max_column_variance(subgoal_stats['eef_pixel_locations'])
        subgoal_stats_cumulative.append(subgoal_stats['var_eef_pixel_locations'])

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))


    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["frontview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    args = parser.parse_args()
    run_trained_agent(args)

