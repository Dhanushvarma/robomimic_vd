import h5py
import torch

import numpy as np

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.tensor_utils as TensorUtils
device =  TorchUtils.get_torch_device(try_to_use_cuda=True)


def read_actions_states(file_path):
    with h5py.File(file_path, 'r') as file:
        print(type(file))
        print(file.keys())
        data_group = file['data']
        print(data_group.keys())
        for demo_name, demo_group in data_group.items():
            print(f"Reading demonstration: {demo_name}")
            print(demo_group.keys())

            # import pdb; pdb.set_trace()


def print_batch_info(batch):
    print("\n============= Batch Info =============")
    for k in batch:
        if k in ["obs", "next_obs"]:
            print("key {}".format(k))
            for obs_key in batch[k]:
                print("    obs key {} with shape {}".format(obs_key, batch[k][obs_key].shape))
        else:
            print("key {} with shape {}".format(k, batch[k].shape))
    print("")



def get_data_loader(dataset_path):
    """
    Get a data loader to sample batches of data.

    Args:
        dataset_path (str): path to the dataset hdf5
    """
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=(                      # observations we want to appear in batches
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "object",
            "human_gaze"
        ),
        dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
            "actions", 
            "rewards", 
            "dones",
        ),
        load_next_obs=True,
        frame_stack=1,
        seq_length=10,                  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="all",          # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,       # can optionally provide a filter key here
    )
    print("\n============= Created Dataset =============")
    print(dataset)
    print("")

    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,       # no custom sampling logic (uniform sampling)
        batch_size=100,     # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=True      # don't provide last batch in dataset pass if it's less than 100 in size
    )

    # import pdb; pdb.set_trace()

    return data_loader

def process_batch_for_training(batch):
    """
    Processes input batch from a data loader to filter out
    relevant information and prepare the batch for training.

    Args:
        batch (dict): dictionary with torch.Tensors sampled
            from a data loader

    Returns:
        input_batch (dict): processed and filtered batch that
            will be used for training 
    """
    input_batch = dict()

    # remove temporal batches for all except scalar signals (to be compatible with model outputs)
    input_batch["obs"] = { k: batch["obs"][k][:, 0, :] for k in batch["obs"] }
    # extract multi-horizon subgoal target
    subgoal_horizon = 10 #TODO(dhanush): move this from here
    input_batch["subgoals"] = {k: batch["next_obs"][k][:, subgoal_horizon - 1, :] for k in batch["next_obs"]}
    input_batch["target_subgoals"] = input_batch["subgoals"] #NOTE(dhanush) : Both the subgoals and the target subgoals are the samething
    input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present

    # we move to device first before float conversion because image observation modalities will be uint8 -
    # this minimizes the amount of data transferred to GPU

    # import pdb; pdb.set_trace()
     #TODO(dhanush) : move this from here
    return TensorUtils.to_float(TensorUtils.to_device(input_batch, device))


def postprocess_batch_for_training(batch, obs_normalization_stats):
        """
        Does some operations (like channel swap, uint8 to float conversion, normalization)
        after @process_batch_for_training is called, in order to ensure these operations
        take place on GPU.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader. Assumed to be on the device where
                training will occur (after @process_batch_for_training
                is called)

            obs_normalization_stats (dict or None): if provided, this should map observation 
                keys to dicts with a "mean" and "std" of shape (1, ...) where ... is the 
                default shape for the observation.

        Returns:
            batch (dict): postproceesed batch
        """

        # ensure obs_normalization_stats are torch Tensors on proper device
        obs_normalization_stats = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(obs_normalization_stats), device))

        # we will search the nested batch dictionary for the following special batch dict keys
        # and apply the processing function to their values (which correspond to observations)
        obs_keys = ["obs", "next_obs", "goal_obs"]

        def recurse_helper(d):
            """
            Apply process_obs_dict to values in nested dictionary d that match a key in obs_keys.
            """
            for k in d:
                if k in obs_keys:
                    # found key - stop search and process observation
                    if d[k] is not None:
                        d[k] = ObsUtils.process_obs_dict(d[k])
                        if obs_normalization_stats is not None:
                            d[k] = ObsUtils.normalize_obs(d[k], obs_normalization_stats=obs_normalization_stats)
                elif isinstance(d[k], dict):
                    # search down into dictionary
                    recurse_helper(d[k])

        recurse_helper(batch)
        return batch


'''
def run_train_loop(data_loader):
    """
    Note: this is a stripped down version of @TrainUtils.run_epoch and the train loop
    in the train function in train.py. Logging and evaluation rollouts were removed.

    Args:
        model (Algo instance): instance of Algo class to use for training
        data_loader (torch.utils.data.DataLoader instance): torch DataLoader for
            sampling batches
    """
    num_epochs = 50
    gradient_steps_per_epoch = 100
    has_printed_batch_info = False

    # ensure model is in train mode
    # model.set_train()

    for epoch in range(1, num_epochs + 1): # epoch numbers start at 1

        # iterator for data_loader - it yields batches
        data_loader_iter = iter(data_loader)

        # record losses
        losses = []

        for _ in range(gradient_steps_per_epoch):

            # load next batch from data loader
            try:
                batch = next(data_loader_iter)
            except StopIteration:
                # data loader ran out of batches - reset and yield first batch
                data_loader_iter = iter(data_loader)
                batch = next(data_loader_iter)

            if not has_printed_batch_info:
                has_printed_batch_info = True
                print_batch_info(batch)

            # process batch for training
            input_batch = process_batch_for_training(batch)
            import pdb; pdb.set_trace()
            # input_batch = postprocess_batch_for_training(input_batch, obs_normalization_stats=None)

            # forward and backward pass
            # info = model.train_on_batch(batch=input_batch, epoch=epoch, validate=False)

            # record loss
            # step_log = model.log_info(info)
            # losses.append(step_log["Loss"])

        # do anything model needs to after finishing epoch
        # model.on_epoch_end(epoch)

        print("Train Epoch {}: Loss {}".format(epoch, np.mean(losses)))







# dataset_path = "/home/dhanush/robomimic_vd/robomimic/scripts/gaze_pred_sketch/low_dim_with_gaze.hdf5"

# # read_actions_states(dataset_path)

# device =  TorchUtils.get_torch_device(try_to_use_cuda=True)

# data_loader = get_data_loader(dataset_path=dataset_path)



# run_train_loop(data_loader=data_loader)

'''