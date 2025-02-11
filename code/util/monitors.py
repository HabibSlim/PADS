"""
Gradient Monitoring class for PyTorch models.
"""

import torch
from typing import Optional, Dict, List, Literal
import logging
import statistics
from collections import defaultdict
import math


class GradientMonitor:
    def __init__(
        self,
        model: torch.nn.Module,
        threshold: float = 1.0,
        window_size: int = 100,
        log_level: int = logging.WARNING,
        norm_type: Literal["L1", "L2"] = "L2",
        break_on_nan: bool = False,  # type: bool
    ):
        """
        Initialize gradient monitoring for a PyTorch model.

        Args:
            model: The PyTorch model to monitor
            threshold: The threshold for what constitutes an unusual gradient (default: 1.0)
            window_size: Number of batches to keep in history for calculating statistics (default: 100)
            log_level: Logging level for gradient warnings (default: logging.WARNING)
            norm_type: Type of norm to use for gradient magnitude ("L1" or "L2")
        """
        self.model = model
        self.threshold = threshold
        self.window_size = window_size
        self.norm_type = norm_type

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Storage for gradient statistics
        self.grad_history: Dict[str, List[float]] = defaultdict(list)
        self.nan_counts: Dict[str, int] = defaultdict(int)
        self.inf_counts: Dict[str, int] = defaultdict(int)
        self.hooks = []
        self.break_on_nan = break_on_nan
        self._register_hooks()

    def _register_hooks(self):
        """Register backward hooks on all parameters of the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Create a closure to capture the current parameter name
                def make_hook(param_name):
                    return lambda grad: self._gradient_hook(grad, param_name)

                hook = param.register_hook(make_hook(name))
                self.hooks.append(hook)

    def _compute_norm(self, grad: torch.Tensor) -> float:
        """Compute the norm of the gradient tensor."""
        if self.norm_type == "L1":
            return torch.norm(grad, p=1).item()
        return torch.norm(grad, p=2).item()

    def _gradient_hook(self, grad: torch.Tensor, param_name: str):
        """Hook function called during backward pass."""
        if grad is None:
            return

        # Check for NaN values
        if torch.isnan(grad).any():
            self.nan_counts[param_name] += 1
            nan_indices = torch.where(torch.isnan(grad))
            self.logger.error(
                f"NaN gradient detected in {param_name}!\n"
                f"NaN count for this parameter: {self.nan_counts[param_name]}\n"
                f"Tensor shape: {grad.shape}\n"
                f"NaN locations: {nan_indices}\n"
                f"Last valid gradient norm: {self.grad_history[param_name][-1] if self.grad_history[param_name] and len(self.grad_history[param_name]) > 0 else 'N/A'}\n"
                f"This indicates a serious numerical issue in training - check for:\n"
                f"1. Learning rate too high\n"
                f"2. Exploding gradients\n"
                f"3. Invalid loss computation\n"
                f"4. Division by zero"
            )
            if self.break_on_nan:
                raise RuntimeError(f"NaN gradient detected in {param_name}")
            return

        # Check for Inf values
        if torch.isinf(grad).any():
            self.inf_counts[param_name] += 1
            inf_indices = torch.where(torch.isinf(grad))
            self.logger.error(
                f"Inf gradient detected in {param_name}!\n"
                f"Inf count for this parameter: {self.inf_counts[param_name]}\n"
                f"Tensor shape: {grad.shape}\n"
                f"Inf locations: {inf_indices}\n"
                f"Last valid gradient norm: {self.grad_history[param_name][-1] if self.grad_history[param_name] else 'N/A'}\n"
                f"This indicates exploding gradients - consider:\n"
                f"1. Gradient clipping\n"
                f"2. Reducing learning rate\n"
                f"3. Batch normalization\n"
                f"4. Better initialization"
            )
            if self.break_on_nan:  # Also break on Inf if configured
                raise RuntimeError(f"Inf gradient detected in {param_name}")
            return

        # Calculate gradient statistics
        try:
            grad_norm = self._compute_norm(grad)

            # Skip if the norm is zero
            if grad_norm == 0:
                return

            self.grad_history[param_name].append(grad_norm)

            # Keep only the recent history
            history = self.grad_history[param_name]
            if len(history) > self.window_size:
                self.grad_history[param_name] = history[-self.window_size :]

            # Check for unusual gradients
            if (
                len(self.grad_history[param_name]) >= 10
            ):  # Need some history for meaningful statistics
                median = statistics.median(self.grad_history[param_name])
                mad = statistics.median(
                    [abs(x - median) for x in self.grad_history[param_name]]
                )

                if mad > 0 and abs(grad_norm - median) / mad > self.threshold:
                    self.logger.warning(
                        f"Unusual gradient detected in {param_name}:\n"
                        f"Current gradient norm ({self.norm_type}): {grad_norm:.4f}\n"
                        f"Median gradient norm: {median:.4f}\n"
                        f"Median absolute deviation: {mad:.4f}\n"
                        f"Deviation from median: {abs(grad_norm - median) / mad:.2f}x threshold"
                    )
        except Exception as e:
            self.logger.error(f"Error processing gradients for {param_name}: {str(e)}")

    def get_nan_inf_counts(self) -> Dict[str, Dict[str, int]]:
        """Get the count of NaN and Inf occurrences for each parameter."""
        return {
            name: {
                "nan_count": self.nan_counts[name],
                "inf_count": self.inf_counts[name],
            }
            for name in set(self.nan_counts.keys()) | set(self.inf_counts.keys())
        }

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get current gradient statistics for all parameters."""
        stats = {}
        for name, history in self.grad_history.items():
            if not history:  # Skip if history is empty
                continue

            if len(history) >= 10:  # Only report if we have enough history
                try:
                    median = statistics.median(history)
                    mad = statistics.median([abs(x - median) for x in history])
                    stats[name] = {
                        "current": history[-1],
                        "median": median,
                        "mad": mad,
                        "min": min(history),
                        "max": max(history),
                        "history_size": len(history),
                    }
                except Exception as e:
                    logging.error(f"Error calculating statistics for {name}: {e}")
                    # Add partial statistics if possible
                    stats[name] = {
                        "current": history[-1] if history else float("nan"),
                        "history_size": len(history),
                    }
        return stats

    def remove_hooks(self):
        """Remove all gradient hooks and clean up internal state."""
        try:
            for hook in self.hooks:
                if hook is not None:  # Check if hook exists
                    try:
                        hook.remove()
                    except Exception as e:
                        # Hook might have been already removed
                        pass
        finally:
            # Always clean up internal state
            self.hooks.clear()
            self.grad_history.clear()
            self.nan_counts.clear()
            self.inf_counts.clear()


class ActivationMonitor:
    def __init__(
        self,
        model: torch.nn.Module,
        threshold: float = 1.0,
        window_size: int = 100,
        log_level: int = logging.WARNING,
        norm_type: Literal["L1", "L2"] = "L2",
        break_on_nan: bool = False,  # type: bool
    ):
        """
        Initialize activation monitoring for a PyTorch model.

        Args:
            model: The PyTorch model to monitor
            threshold: The threshold for what constitutes an unusual activation
            window_size: Number of batches to keep in history
            log_level: Logging level for warnings
            norm_type: Type of norm to use for activation magnitude
            break_on_nan: Whether to raise an exception when NaN is detected
        """
        self.model = model
        self.threshold = threshold
        self.window_size = window_size
        self.norm_type = norm_type
        self.break_on_nan = break_on_nan

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Storage for activation statistics
        self.activation_history: Dict[str, List[float]] = defaultdict(list)
        self.nan_counts: Dict[str, int] = defaultdict(int)
        self.inf_counts: Dict[str, int] = defaultdict(int)
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on all modules of the model."""
        for name, module in self.model.named_modules():
            # Skip container modules (Sequential, ModuleList, etc.)
            if isinstance(
                module, (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict)
            ):
                continue

            # Skip modules that are just containers
            if not hasattr(module, "forward"):
                continue

            def make_hook(module_name):
                def hook(module, input_tensor, output_tensor):
                    return self._forward_hook(module_name, input_tensor, output_tensor)

                return hook

            hook = module.register_forward_hook(make_hook(name))
            self.hooks.append(hook)

    def _compute_norm(self, tensor: torch.Tensor) -> float:
        """Compute the norm of the tensor."""
        if self.norm_type == "L1":
            return torch.norm(tensor, p=1).item()
        return torch.norm(tensor, p=2).item()

    def _forward_hook(
        self, module_name: str, input_tensor: tuple, output_tensor: torch.Tensor
    ):
        """Hook function called during forward pass."""
        # Handle different types of outputs
        if isinstance(output_tensor, (tuple, list)):
            if not output_tensor:  # Empty sequence
                return
            output_tensor = output_tensor[0]  # Take first element
        elif isinstance(output_tensor, dict):
            if not output_tensor:  # Empty dict
                return
            # Take the first tensor value
            output_tensor = next(
                (v for v in output_tensor.values() if isinstance(v, torch.Tensor)), None
            )
            if output_tensor is None:
                return

        if not isinstance(output_tensor, torch.Tensor):
            return

        # Check for NaN values in output
        if torch.isnan(output_tensor).any():
            self.nan_counts[module_name] += 1
            nan_indices = torch.where(torch.isnan(output_tensor))

            # Check input tensors for NaN as well
            input_nan = any(
                torch.isnan(x).any()
                for x in input_tensor
                if isinstance(x, torch.Tensor)
            )

            self.logger.error(
                f"NaN activation detected in module {module_name}!\n"
                f"NaN count for this module: {self.nan_counts[module_name]}\n"
                f"Output tensor shape: {output_tensor.shape}\n"
                f"NaN locations in output: {nan_indices}\n"
                f"Input also contains NaN: {input_nan}\n"
                f"Last valid activation norm: {self.activation_history[module_name][-1] if self.activation_history[module_name] and len(self.activation_history[module_name]) > 0 else 'N/A'}\n"
                f"This indicates a numerical issue - check for:\n"
                f"1. Division by zero\n"
                f"2. Log of zero/negative numbers\n"
                f"3. Overflow in exp/softmax\n"
                f"4. Invalid input normalization"
            )
            if self.break_on_nan:
                raise RuntimeError(f"NaN activation detected in module {module_name}")
            return

        # Check for Inf values
        if torch.isinf(output_tensor).any():
            self.inf_counts[module_name] += 1
            inf_indices = torch.where(torch.isinf(output_tensor))
            self.logger.error(
                f"Inf activation detected in module {module_name}!\n"
                f"Inf count for this module: {self.inf_counts[module_name]}\n"
                f"Tensor shape: {output_tensor.shape}\n"
                f"Inf locations: {inf_indices}\n"
                f"Last valid activation norm: {self.activation_history[module_name][-1] if self.activation_history[module_name] and len(self.activation_history[module_name]) > 0 else 'N/A'}\n"
                f"This indicates a numerical overflow - check for:\n"
                f"1. Very large weights\n"
                f"2. Unstable activation functions\n"
                f"3. Incorrect input scaling\n"
                f"4. Missing normalization layers"
            )
            if self.break_on_nan:  # Also break on Inf if configured
                raise RuntimeError(f"Inf activation detected in module {module_name}")
            return

        # Calculate activation statistics
        try:
            # Compute norm with memory efficiency
            with torch.no_grad():
                activation_norm = self._compute_norm(output_tensor.detach())
            torch.cuda.empty_cache()  # Clear any temporary tensors

            # Skip if the norm is zero
            if activation_norm == 0:
                return

            # Update history with thread-safe operation
            history = self.activation_history[module_name]
            history.append(activation_norm)
            if len(history) > self.window_size:
                self.activation_history[module_name] = history[-self.window_size :]

            # Check for unusual activations
            if len(history) >= 10:
                median = statistics.median(history)
                mad = statistics.median([abs(x - median) for x in history])

                if mad > 0 and abs(activation_norm - median) / mad > self.threshold:
                    self.logger.warning(
                        f"Unusual activation detected in module {module_name}:\n"
                        f"Current activation norm ({self.norm_type}): {activation_norm:.4f}\n"
                        f"Median activation norm: {median:.4f}\n"
                        f"Median absolute deviation: {mad:.4f}\n"
                        f"Deviation from median: {abs(activation_norm - median) / mad:.2f}x threshold"
                    )
        except Exception as e:
            self.logger.error(
                f"Error processing activations for {module_name}: {str(e)}"
            )

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get current activation statistics for all modules."""
        stats = {}
        for name, history in self.activation_history.items():
            if not history:
                continue

            if len(history) >= 10:
                try:
                    median = statistics.median(history)
                    mad = statistics.median([abs(x - median) for x in history])
                    stats[name] = {
                        "current": history[-1],
                        "median": median,
                        "mad": mad,
                        "min": min(history),
                        "max": max(history),
                        "history_size": len(history),
                    }
                except Exception as e:
                    self.logger.error(f"Error calculating statistics for {name}: {e}")
                    stats[name] = {
                        "current": history[-1] if history else float("nan"),
                        "history_size": len(history),
                    }
        return stats

    def get_nan_inf_counts(self) -> Dict[str, Dict[str, int]]:
        """Get the count of NaN and Inf occurrences for each module."""
        return {
            name: {
                "nan_count": self.nan_counts[name],
                "inf_count": self.inf_counts[name],
            }
            for name in set(self.nan_counts.keys()) | set(self.inf_counts.keys())
        }

    def remove_hooks(self):
        """Remove all forward hooks and clean up internal state."""
        try:
            for hook in self.hooks:
                if hook is not None:
                    try:
                        hook.remove()
                    except Exception:
                        pass
        finally:
            # Always clean up internal state
            self.hooks.clear()
            self.activation_history.clear()
            self.nan_counts.clear()
            self.inf_counts.clear()


# Example usage
def example_usage__forward():
    # Initialize model and move to appropriate device
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize monitor with appropriate settings
    monitor = ActivationMonitor(
        model,
        threshold=2.0,
        norm_type="L2",
        break_on_nan=True,
        log_level=logging.INFO,  # Adjust based on needs
    )

    try:
        for epoch in range(num_epochs):
            model.train()  # Ensure training mode
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Move batch to device
                    inputs = batch.input.to(device)
                    targets = batch.target.to(device)

                    # Forward pass
                    outputs = model(inputs)

                    # Rest of training loop...
                    loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                except RuntimeError as e:
                    if "NaN activation detected" in str(e):
                        logging.error("Forward pass stopped due to NaN activations")
                        nan_inf_counts = monitor.get_nan_inf_counts()
                        logging.error(f"NaN/Inf statistics: {nan_inf_counts}")

                        # Get last known good statistics
                        stats = monitor.get_statistics()
                        logging.error(f"Last known good statistics: {stats}")

                        # Optionally save model state for debugging
                        torch.save(
                            {
                                "epoch": epoch,
                                "batch_idx": batch_idx,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "nan_inf_counts": nan_inf_counts,
                                "last_stats": stats,
                            },
                            f"debug_checkpoint_epoch{epoch}_batch{batch_idx}.pt",
                        )

                        raise
                except Exception as e:
                    logging.error(f"Unexpected error during training: {str(e)}")
                    raise

            # Epoch-end statistics
            stats = monitor.get_statistics()
            nan_inf_counts = monitor.get_nan_inf_counts()

            logging.info(f"Epoch {epoch} activation statistics: {stats}")
            if any(
                counts["nan_count"] > 0 or counts["inf_count"] > 0
                for counts in nan_inf_counts.values()
            ):
                logging.warning(f"NaN/Inf activations occurred in epoch {epoch}")

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise
    finally:
        # Always clean up
        monitor.remove_hooks()
        torch.cuda.empty_cache()


# Example usage:
def example_usage__backward():
    model = torch.nn.Linear(10, 1)
    monitor = GradientMonitor(
        model, threshold=2.0, norm_type="L2", break_on_nan=True  # Stop on NaN gradients
    )

    # In your training loop:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    try:
        for epoch in range(num_epochs):
            for batch in dataloader:
                try:
                    optimizer.zero_grad()
                    output = model(batch.input)
                    loss = criterion(output, batch.target)
                    loss.backward()

                    optimizer.step()
                except RuntimeError as e:
                    if "NaN gradient detected" in str(e):
                        print("Training stopped due to NaN gradients")
                        # Check which parameters had NaN issues
                        nan_inf_counts = monitor.get_nan_inf_counts()
                        print("NaN/Inf statistics:", nan_inf_counts)
                        raise  # Re-raise to stop training

            # Check statistics at epoch end
            stats = monitor.get_statistics()
            nan_inf_counts = monitor.get_nan_inf_counts()
            print(f"Epoch {epoch} gradient statistics:", stats)
            if any(
                counts["nan_count"] > 0 or counts["inf_count"] > 0
                for counts in nan_inf_counts.values()
            ):
                print("Warning: NaN/Inf gradients occurred in this epoch")

    finally:
        # Always remove hooks when done
        monitor.remove_hooks()
