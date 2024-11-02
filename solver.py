from model import Generator, Discriminator
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime


class Solver(object):
    """Solver for testing StarGAN with pre-trained models."""

    def __init__(self, config):
        """Initialize configurations."""

        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.g_repeat_num = config.g_repeat_num

        # Device configuration.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Build the model.
        self.build_model()

    def build_model(self):
        """Create a generator."""
        self.G = Generator(conv_dim=self.g_conv_dim, c_dim=self.c_dim, repeat_num=self.g_repeat_num).to(self.device)
        print("Generator initialized.")

    def restore_model(self, G_checkpoint):
        """Restore the trained generator."""
        G_path = os.path.join(self.model_save_dir, G_checkpoint)
        if not os.path.exists(G_path):
            raise FileNotFoundError(f"Generator checkpoint not found at {G_path}")
        self.G.load_state_dict(torch.load(G_path, map_location=self.device))
        self.G.eval()
        print(f"Loaded Generator from {G_path}")

    def test(self, image_tensor, c_trg):
        """Translate image using the trained generator."""
        with torch.no_grad():
            x_fake = self.G(image_tensor, c_trg)
        return x_fake