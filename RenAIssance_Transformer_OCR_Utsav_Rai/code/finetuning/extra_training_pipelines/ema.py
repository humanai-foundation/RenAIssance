import torch
from copy import deepcopy

class ExponentialMovingAverage:
    """
    Maintains moving averages of model parameters using exponential decay.
    
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be maintained.
        decay (float): The decay rate for the EMA. Higher values (closer to 1.0)
                      result in slower decay and smoother averaging.
        device (torch.device): Device where the EMA parameters will be stored.
    """
    def __init__(self, model, decay=0.9999, device=None):
        self.model = model
        self.decay = decay
        self.device = device if device else next(model.parameters()).device
        self.shadow_params = [
            deepcopy(p.data).to(device) for p in model.parameters()
        ]
        self.collected_params = []
        self.num_updates = 0
        
    def update(self, decay=None):
        """
        Update the moving averages based on the latest model parameters.
        
        Args:
            decay (float, optional): Override the decay rate for this update.
        """
        decay = decay if decay is not None else self.decay
        
        # Increase effective decay rate as training progresses for better stability
        if self.num_updates > 0:
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
            
        self.num_updates += 1
        
        for i, param in enumerate(self.model.parameters()):
            if param.requires_grad:
                self.shadow_params[i].sub_((1.0 - decay) * (self.shadow_params[i] - param.data))
    
    def copy_to(self, model=None):
        """
        Copy the EMA parameters to the provided model or the original model.
        
        Args:
            model (torch.nn.Module, optional): Model to copy parameters to.
                                              If None, uses the original model.
        
        Returns:
            The model with EMA parameters.
        """
        if model is None:
            model = self.model
        
        # Store current parameters for later restoration
        self.collected_params = [param.data.clone() for param in model.parameters()]
        
        # Copy EMA parameters to model
        for i, param in enumerate(model.parameters()):
            if param.requires_grad:
                param.data.copy_(self.shadow_params[i])
                
        return model
    
    def restore(self, model=None):
        """
        Restore the original parameters that were replaced by copy_to().
        
        Args:
            model (torch.nn.Module, optional): Model to restore parameters to.
                                              If None, uses the original model.
        
        Returns:
            The model with restored parameters.
        """
        if model is None:
            model = self.model
            
        # Restore original parameters
        for i, param in enumerate(model.parameters()):
            if param.requires_grad and i < len(self.collected_params):
                param.data.copy_(self.collected_params[i])
                
        self.collected_params = []
        return model