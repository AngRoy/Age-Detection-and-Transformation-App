# logger.py

from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """TensorBoard logger using PyTorch's built-in support."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        self.writer.add_scalar(tag, value, step)

    def close(self):
        """Close the summary writer."""
        self.writer.close()
