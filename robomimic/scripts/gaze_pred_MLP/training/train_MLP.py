import argparse
import torch
import torch.optim as optim
import sys
import wandb
import os
import robomimic.scripts.gaze_pred_MLP.utils.common_utils as CU
from robomimic.scripts.gaze_pred_MLP.models.MLP import GaussianMLP
from torch.utils.data import DataLoader
from robomimic.scripts.gaze_pred_MLP.utils.data_loader import *


class MLP_trainer:

    def __init__(self, cfg):

        self.cfg = cfg # Config from YAML file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GaussianMLP(cfg['input_dim'], cfg['hidden_dim'], cfg['output_dim']).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg['learning_rate'])

        # Loading the model onto device
        self.model.to(self.device)

        #WandB logging
        self.wandb = CU.initialize_wandb(cfg["wandb_project"], cfg["wandb_entity"])

        # File path for saving stuff (runs_cvae/run_name/ - 1)checkpoints, 2)model, 3)yaml file from cfg)
        self.cpt_path = "./runs_MLP/" + cfg['run_name'] + '/'
        self.checkpoints_dir = os.path.join(self.cpt_path, 'checkpoints/')
        self.model_file = os.path.join(self.cpt_path, 'model.pth')
        self.config_file = os.path.join(self.cpt_path, 'config.yaml')

        # Initialize directories and files based on resume flag
        CU.initialize_directories_and_files(self.cpt_path, self.checkpoints_dir, self.model_file, self.config_file, self.cfg,resume=False)
    

        # Stuff needed for the Train Loop
        self.data_loader = get_data_loader(self.cfg['dataset_path'])
        self.num_epochs = self.cfg['num_epochs']
        self.gradient_steps_per_epoch = self.cfg['gradient_steps_per_epoch']



    def train(self):

        has_printed_batch_info = False

        for epoch in range(1, self.num_epochs + 1):
            data_loader_iter = iter(self.data_loader)
            losses = []

            for _ in range(self.gradient_steps_per_epoch):
                try:
                    batch = next(data_loader_iter)
                except StopIteration:
                    data_loader_iter = iter(self.data_loader)
                    batch = next(data_loader_iter)

                if not has_printed_batch_info:
                    has_printed_batch_info = True
                    print_batch_info(batch) # NOTE(dhanush) : This is Robomimic code

                input_batch = process_batch_for_training(batch) # NOTE(dhanush) : This is Robomimic code
                # NOTE(dhanush) : input_batch.keys() : dict_keys(['obs', 'subgoals', 'target_subgoals', 'goal_obs'])
                # input_batch['obs'].keys()  --> dict_keys(['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object', 'human_gaze'])
                # input_batch['target_subgoals'].keys()  -->  dict_keys(['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object', 'human_gaze']) 

                # Assuming input_batch is a dictionary containing 'obs' and 'target_subgoals' components
                obs = input_batch['obs']
                target_subgoals = input_batch['target_subgoals']

                # Extract the relevant components for the MLP Input
                robot0_eef_pos_obs = obs['robot0_eef_pos']
                robot0_eef_quat_obs = obs['robot0_eef_quat']
                robot0_gripper_qpos_obs = obs['robot0_gripper_qpos']
                object_obs = obs['object']

                robot0_eef_pos_target = target_subgoals['robot0_eef_pos']
                robot0_eef_quat_target = target_subgoals['robot0_eef_quat']
                robot0_gripper_qpos_target = target_subgoals['robot0_gripper_qpos']
                object_target = target_subgoals['object']

                # Extract the relevant component for the MLP Target
                target_x = input_batch['obs']['human_gaze'][:, 0]  # Extract the x-component
                target_y = input_batch['obs']['human_gaze'][:, 1]  # Extract the y-component

                # Assuming you have extracted the relevant components as shown in your code
                obs_tensors = [
                    robot0_eef_pos_obs, robot0_eef_quat_obs,
                    robot0_gripper_qpos_obs, object_obs
                ]

                target_subgoals_tensors = [
                    robot0_eef_pos_target, robot0_eef_quat_target,
                    robot0_gripper_qpos_target, object_target
                ]


                obs_concatenated = torch.cat(obs_tensors, dim=1)
                target_subgoals_concatenated = torch.cat(target_subgoals_tensors, dim=1)
                input_data = torch.cat([obs_concatenated, target_subgoals_concatenated], dim=1) #NOTE(dhanush) : This is the input to the MLP

                # Forward pass
                mean_x, std_x, mean_y, std_y = self.model(input_data)

                # import pdb; pdb.set_trace()

                # Calculate the negative log likelihood loss for each dimension separately
                loss = CU.negative_log_likelihood_loss(target_x, mean_x[:, 0], std_x[:, 0], target_y, mean_y[:, 1], std_y[:, 1]) 

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

                # Log training batch metrics to wandb
                wandb.log({"train_loss": loss.item(), "epoch": epoch})


            avg_loss = np.mean(losses)

            # Log average training loss to wandb
            wandb.log({"average_train_loss": avg_loss, "epoch": epoch})

            # Optionally, print or log other metrics here

            if epoch % self.cfg['log_frequency'] == 0:
                print(f"Epoch {epoch}: Average Loss: {avg_loss:.4f}")
                self.save_checkpoint(epoch=epoch)

        # Save the final model
        torch.save(self.model.state_dict(), self.model_file)
        print(f"Model saved at {self.model_file}")


    def save_checkpoint(self, epoch):

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cfg': self.cfg
        }

        checkpoint_path = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    

    def print_parameter_summary(self):
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

        print("\nOptimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            print(var_name, "\t", self.optimizer.state_dict()[var_name])


    #TODO: write code for hyperparameter tuning
            
        

def main(config_path):
    # Load configuration
    cfg = CU.load_config(config_path)

    # Initialize the trainer
    trainer = MLP_trainer(cfg)

    # Train
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--resume", type=lambda x: (str(x).lower() == 'true'), required=False, default=False,
                        help="Flag to resume training from the last checkpoint.")
    args = parser.parse_args()

    main(args.config)












            