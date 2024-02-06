import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from model import negative_log_likelihood_loss, GaussianMLP
import wandb



class GaussianMLPTrainer:
    def __init__(self, model, learning_rate=0.001, project_name="your_project_name"):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.project_name = project_name

        # self.


    def train(self, train_loader, num_epochs):
        wandb.init(project=self.project_name)
        wandb.watch(self.model)

        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in train_loader:
                inputs, targets_x, targets_y = batch

                # Flatten the input vectors and concatenate them
                input_data = torch.cat(inputs, dim=1)

                # Forward pass
                mean_x, std_x, mean_y, std_y = self.model(input_data)

                # Calculate negative log likelihood loss separately for gaze_x and gaze_y
                loss = negative_log_likelihood_loss(targets_x, mean_x, std_x, targets_y, mean_y, std_y)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

            # Log metrics to wandb
            wandb.log({"Epoch": epoch + 1, "Loss": avg_loss})

        # Finish the wandb run
        wandb.finish()

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()




if __name__ == "__main__":

    wandb.init(project="your_project_name")

    model = GaussianMLP(input_dim=4, hidden_dim=1024, output_dim=2)

    trainer = GaussianMLPTrainer(model, learning_rate=0.001, project_name="your_project_name")



