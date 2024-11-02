# test_stargan.py

import argparse
import os
from model import Generator
import torch
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms


def get_test_loader(image_path, image_size):
    """
    Creates a DataLoader for a single image.

    Parameters:
        image_path (str): Path to the input image.
        image_size (int): Desired image size after transformation.

    Returns:
        image_tensor (torch.Tensor): Transformed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size, Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def main(config):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Generator
    G = Generator(conv_dim=config.g_conv_dim, c_dim=config.c_dim, repeat_num=config.g_repeat_num).to(device)

    # Load pre-trained Generator
    G_path = os.path.join(config.model_save_dir, config.G_checkpoint)
    if not os.path.exists(G_path):
        raise FileNotFoundError(f"Generator checkpoint not found at {G_path}")

    G.load_state_dict(torch.load(G_path, map_location=device))
    G.eval()
    print(f"Loaded Generator from {G_path}")

    # Prepare input image
    input_image = config.input_image
    if not os.path.exists(input_image):
        raise FileNotFoundError(f"Input image not found at {input_image}")

    image_tensor = get_test_loader(input_image, config.image_size).to(device)

    # Prepare target attribute vector
    # Assuming binary attributes; adjust as per your model's requirements
    # Example for CelebA: [Black_Hair, Blond_Hair, Brown_Hair, Male, Young]
    # Index of 'Young' attribute is 4
    c_dim = config.c_dim
    c_trg = torch.zeros((1, c_dim)).to(device)
    if config.transform_choice == 'older':
        c_trg[0, 4] = 0  # Not Young
    elif config.transform_choice == 'younger':
        c_trg[0, 4] = 1  # Young
    else:
        raise ValueError("Invalid transform_choice. Choose 'older' or 'younger'.")

    # Generate fake image
    with torch.no_grad():
        x_fake = G(image_tensor, c_trg)

    # Denormalize and save the transformed image
    x_fake = (x_fake + 1) / 2  # Denormalize from [-1,1] to [0,1]
    transformed_image_path = os.path.join(config.result_dir, 'transformed_image.jpg')
    save_image(x_fake, transformed_image_path)
    print(f"Transformed image saved at {transformed_image_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StarGAN Image Transformation")
    
    # Transformation configuration
    parser.add_argument('--transform_choice', type=str, default='older', choices=['older', 'younger'],
                        help="Transformation choice: 'older' or 'younger'")
    
    # Model configuration
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    
    # Paths
    parser.add_argument('--model_save_dir', type=str, default='stargan_celeba_256/models',
                        help='directory to load the pre-trained Generator model')
    parser.add_argument('--G_checkpoint', type=str, default='200000-G.ckpt',
                        help='Generator checkpoint filename')
    parser.add_argument('--input_image', type=str, required=True,
                        help='path to the input image for transformation')
    parser.add_argument('--result_dir', type=str, default='stargan_celeba_256/results',
                        help='directory to save the transformed image')
    
    args = parser.parse_args()
    
    # Create result directory if not exists
    os.makedirs(args.result_dir, exist_ok=True)
    
    main(args)
