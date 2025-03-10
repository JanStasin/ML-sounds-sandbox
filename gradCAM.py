import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define a GradCAM helper class
class gradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.model.eval()  # Make sure the model is in evaluation mode
        self.target_layer = target_layer
        
        # Initialize containers for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks to capture the forward activation and backward gradients
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; we extract the first element which is the gradients
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> torch.Tensor:
        """Computes the Grad-CAM heatmap for given input and target class.
        
        Args:
            input_tensor (torch.Tensor): A tensor of shape (B, C, H, W) representing your input.
            target_class (int, optional): The target class index. If None, uses the model's predicted class.
            
        Returns:
            torch.Tensor: The generated heatmap of shape (B, 1, H, W) normalized [0, 1].
        """
        # Forward pass
        output = self.model(input_tensor)
        if target_class is None:
            # Pick the top predicted class for the first (and assumed only) sample.
            target_class = output.argmax(dim=1)[0].item()

        # Create one-hot encoding for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        
        # Backward pass: compute gradients for the target class
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute the weights - average the gradients across spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # shape: (B, channels, 1, 1)
        
        # Compute the weighted combination of forward activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # Apply ReLU to ensure only positive influences
        cam = F.relu(cam)
        
        # Normalize the CAM so that its values lie between 0 and 1
        b, c, h, w = cam.shape
        cam = cam.view(b, -1)
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam = cam.view(b, 1, h, w)
        
        # Optionally, you can interpolate the heatmap to match the input size.
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        return cam, target_class

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()