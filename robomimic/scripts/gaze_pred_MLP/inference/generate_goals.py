from robomimic.scripts.gaze_pred_MLP.models.MLP import GaussianMLP
import torch

###
input_dim = 32
latent_dim = 128
output_dim = 32


###
def generate_goal(model, current_state, device):
    # Sample a latent vector from a standard Gaussian distribution
    z = torch.randn(1, latent_dim).to(device)

    # Ensure the current state is in the correct format and on the correct device
    current_state = current_state.float().to(device)

    # Generate a goal prediction using the decoder
    with torch.no_grad():  # Ensure gradients are not computed
        predicted_goal = model.generate(z, current_state)

    return predicted_goal


model = GaussianMLP(input_dim, latent_dim, output_dim)
model.load_state_dict(torch.load('MLP_model.pth'))
model.eval()  # Set the model to evaluation mode

# current_state = torch.tensor([5.3955e-03, 2.9496e+00, 7.6878e-01, 2.0833e-02, -2.0833e-02,
#                               -1.6002e-02, 9.2226e-03, 8.3162e-01, -2.8588e-01, 0.0000e+00,
#                               0.0000e+00, 9.5826e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#                               0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#                               0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#                               0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#                               0.0000e+00, 0.0000e+00])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# predicted_goal = generate_goal(model, current_state, device)
# print("Predicted Goal:", predicted_goal.cpu().numpy())
