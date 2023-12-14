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

import robomimic
import robomimic.utils.camera_utils as CameraUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy, RolloutPolicy_HBC


def plot_array(data, title):
    """
    Plot a given numpy array of size (n,) as a line plot.

    Args:
        data (np.array): A numpy array of size (n,).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data, linestyle='-', marker='o', color='b')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('Subgoal Call Index')
    plt.ylabel('Variance')
    plt.show()


def calculate_max_column_variance(subgoal_data):
    """
    Calculate the maximum column-wise variance for each (10, 3) array in the provided list of arrays.

    Args:
        subgoal_data (list of np.array): A list where each element is a (10, 3) numpy array.

    Returns:
        np.array: An array of maximum variances. Each element corresponds to the maximum of the three column variances
                  of one (10, 3) array.
    """
    max_variances = [np.max(np.var(array, axis=0)) for array in subgoal_data]

    return np.array(max_variances)



def extract_world_from_subgoals(subgoal_samples_dict):
    """
    Extracts the position of the end-effector (eef) of robot0 from a dictionary of subgoal samples
    and converts it to a NumPy array in the desired shape.

    Args:
        subgoal_samples_dict (dict): A dictionary containing various subgoal samples. It should
                                     include the key 'robot0_eef_pos', which is a PyTorch tensor.

    Returns:
        numpy.ndarray: A NumPy array containing the end-effector positions, reshaped to [..., 3].

    Raises:
        KeyError: If 'robot0_eef_pos' is not a key in the input dictionary.
        TypeError: If the value associated with 'robot0_eef_pos' is not a PyTorch tensor.
    """
    try:
        # Ensure 'robot0_eef_pos' is in the dictionary
        eef_pos_tensor = subgoal_samples_dict['robot0_eef_pos']

        # Check if it's a PyTorch tensor
        if not isinstance(eef_pos_tensor, torch.Tensor):
            raise TypeError("The 'robot0_eef_pos' entry must be a PyTorch tensor.")

        # Move tensor to CPU, detach from the computation graph, convert to NumPy array, and reshape
        return eef_pos_tensor.cpu().detach().numpy().reshape(-1, 3)

    except KeyError:
        raise KeyError("The key 'robot0_eef_pos' was not found in the input dictionary.")



def choose_subgoal(sg_proposals, choose_subgoal_index):
    """
    Selects a specific subgoal for each key in the provided proposals based on the given index.

    Args:
        sg_proposals (dict): A dictionary containing subgoal proposals, where each key maps to a list of tensors.
        choose_subgoal_index (int): The index of the subgoal to be selected for each key.

    Returns:
        dict: A dictionary with the chosen subgoal for each key.

    Raises:
        IndexError: If the `choose_subgoal_index` is out of range for any of the subgoal lists.
    """
    chosen_subgoal = {}

    for key in sg_proposals:
        if not (0 <= choose_subgoal_index < len(sg_proposals[key][0])):
            raise IndexError(f"Index {choose_subgoal_index} is out of range for key '{key}'.")

        chosen_subgoal[key] = sg_proposals[key][0][choose_subgoal_index].unsqueeze(0)

    return chosen_subgoal


def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None):
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
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    ### We obtain the camera trnasformation matrix for the env
    camera_transformation_matrix = env.get_camera_transform_matrix(camera_names[0], 2048, 2048)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    subgoal_stats = dict( eef_world_coordinates=[], eef_pixel_locations=[])

    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))

    try:
        for step_i in range(0,horizon): #TODO: Check if this is fine

            print(step_i)
            interval = 50  # Subgoal Update Rate

            # Check if time to update subgoal
            if step_i % interval == 0:

                # get subgoal samples from planner (VAE)
                sg_proposals = policy.subgoal_proposals(ob=obs)

                print("The shape of the subgoal proposal is", sg_proposals['robot0_eef_pos'].shape)

                world_points = extract_world_from_subgoals(sg_proposals)

                subgoal_stats['eef_world_coordinates'].append(world_points) # Tracking the world cordinates proposals

                pp_sgs = env.project_points_from_world_to_camera(world_points, camera_transformation_matrix, 2048, 2048)

                subgoal_stats['eef_pixel_locations'].append(pp_sgs) # Tracking the world cordinates proposals

                choose_subgoal_index = int(input("Choose one of them, using the index"))

                choosen_subgoal = choose_subgoal(sg_proposals, choose_subgoal_index)

                ### Setting the subgoal for the actor
                policy.set_subgoal(choosen_subgoal)


            # print("Subgoal Being used is", choosen_subgoal['robot0_gripper_qpos'])
            act = policy(ob=obs, goal=None)

            # play action
            next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0], height=2048, width=2048)
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=2048, width=2048, camera_name=cam_name))
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



    return stats, traj, subgoal_stats


def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
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

    rollout_stats = []
    for i in range(rollout_num_episodes):
        stats, traj, subgoal_stats = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names, 
        )
        rollout_stats.append(stats)
        variance_world_coordiantes = calculate_max_column_variance(subgoal_stats['eef_world_coordinates'])
        variance_pixel_locations = calculate_max_column_variance(subgoal_stats['eef_pixel_locations'])

        plot_array(variance_pixel_locations, title="Variance in pixel locations")
        plot_array(variance_world_coordiantes, title="Variance in world coordinates")


        pdb.set_trace()


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

