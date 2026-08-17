import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from model import UNetGenerator


class Pix2PixTester:
    def __init__(self, checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the tester with a trained model
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to run inference on
        """
        self.device = device
        
        # Initialize generator
        self.generator = UNetGenerator().to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Using device: {device}")
    
    def denormalize(self, tensor):
        """Denormalize tensor from [-1, 1] to [0, 1]"""
        return tensor * 0.5 + 0.5
    
    def load_and_preprocess(self, image_path):
        """Load and preprocess an image"""
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image)
        return image, tensor.unsqueeze(0)  # Add batch dimension
    
    def generate_frontal(self, side_image_path):
        """
        Generate frontal profile from side profile
        
        Args:
            side_image_path: Path to side profile image
            
        Returns:
            original_image: Original PIL image
            generated_tensor: Generated image tensor
        """
        original_img, side_tensor = self.load_and_preprocess(side_image_path)
        
        with torch.no_grad():
            side_tensor = side_tensor.to(self.device)
            generated = self.generator(side_tensor)
        
        return original_img, generated
    
    def save_result(self, original_img, generated_tensor, output_path, show_comparison=True):
        """
        Save the generated result
        
        Args:
            original_img: Original side profile image
            generated_tensor: Generated frontal profile tensor
            output_path: Path to save the result
            show_comparison: Whether to save comparison image
        """
        # Denormalize and convert to numpy
        generated_np = self.denormalize(generated_tensor[0]).cpu().numpy().transpose(1, 2, 0)
        generated_np = np.clip(generated_np, 0, 1)
        
        # Convert to PIL Image
        generated_pil = Image.fromarray((generated_np * 255).astype(np.uint8))
        
        if show_comparison:
            # Create comparison figure
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original side profile
            axes[0].imshow(original_img)
            axes[0].set_title('Input: Side Profile', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Generated frontal profile
            axes[1].imshow(generated_pil)
            axes[1].set_title('Generated: Frontal Profile', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Comparison saved to: {output_path}")
            
            # Also save the generated image separately
            generated_only_path = output_path.replace('.png', '_generated.png')
            generated_pil.save(generated_only_path)
            print(f"Generated image saved to: {generated_only_path}")
        else:
            generated_pil.save(output_path)
            print(f"Generated image saved to: {output_path}")
        
        return generated_pil
    
    def test_single_image(self, image_path, output_dir='output', prefix='test'):
        """
        Test a single image and save result
        
        Args:
            image_path: Path to input side profile image
            output_dir: Directory to save output
            prefix: Prefix for output filename
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nProcessing: {image_path}")
        
        # Generate frontal profile
        original_img, generated = self.generate_frontal(image_path)
        
        # Create output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f'{prefix}_{base_name}_comparison.png')
        
        # Save result
        self.save_result(original_img, generated, output_path, show_comparison=True)
        
        return generated
    
    def test_folder(self, input_folder, output_dir='output'):
        """
        Test all images in a folder
        
        Args:
            input_folder: Folder containing test images
            output_dir: Directory to save outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        
        if not image_files:
            print(f"No images found in {input_folder}")
            return
        
        print(f"\nFound {len(image_files)} images to process")
        
        for image_path in image_files:
            self.test_single_image(image_path, output_dir, prefix='result')
        
        print(f"\nAll results saved to: {output_dir}")


def test_with_multiple_checkpoints(image_path, checkpoint_dir='checkpoints', output_dir='output'):
    """
    Test an image with multiple checkpoints to compare results
    
    Args:
        image_path: Path to input image
        checkpoint_dir: Directory containing checkpoints
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all checkpoints
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pth')))
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Load original image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    original_img = Image.open(image_path).convert('RGB')
    
    results = []
    checkpoint_names = []
    
    # Test with selected checkpoints (to avoid too many)
    selected_checkpoints = [checkpoints[i] for i in range(0, len(checkpoints), max(1, len(checkpoints)//5))]
    selected_checkpoints.append(checkpoints[-1])  # Always include the final one
    
    for checkpoint_path in selected_checkpoints:
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pth', '')
        checkpoint_names.append(checkpoint_name)
        
        print(f"Testing with {checkpoint_name}...")
        
        tester = Pix2PixTester(checkpoint_path)
        _, generated = tester.generate_frontal(image_path)
        
        generated_np = tester.denormalize(generated[0]).cpu().numpy().transpose(1, 2, 0)
        generated_np = np.clip(generated_np, 0, 1)
        results.append(generated_np)
    
    # Create comparison figure
    num_results = len(results)
    fig, axes = plt.subplots(1, num_results + 1, figsize=(4 * (num_results + 1), 4))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Input\n(Side Profile)', fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Generated images
    for i, (result, name) in enumerate(zip(results, checkpoint_names)):
        axes[i + 1].imshow(result)
        axes[i + 1].set_title(f'{name}\n(Generated)', fontsize=10)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'checkpoint_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nCheckpoint comparison saved to: {output_path}")


def main():
    """Main testing function"""
    
    # ============= CONFIGURATION - EDIT THESE =============
    CHECKPOINT_PATH = 'checkpoints/pix2pix_epoch_70.pth'  # Model to use
    TEST_IMAGE = 'test/test-1.jpeg'                     # Input image
    OUTPUT_DIR = 'output'                               # Output folder
    # ======================================================
    
    print("="*60)
    print("Pix2Pix cGAN - Testing Script")
    print("="*60)
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("\nAvailable checkpoints:")
        checkpoints = glob.glob('checkpoints/*.pth')
        for cp in checkpoints:
            print(f"  - {cp}")
        return
    
    # Check if test image exists
    if not os.path.exists(TEST_IMAGE):
        print(f"Error: Test image not found at {TEST_IMAGE}")
        print("\nAvailable test images:")
        test_images = glob.glob('test/*.*')
        for img in test_images:
            print(f"  - {img}")
        return
    
    print(f"\nModel: {CHECKPOINT_PATH}")
    print(f"Input: {TEST_IMAGE}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Initialize tester
    tester = Pix2PixTester(CHECKPOINT_PATH)
    
    # Generate frontal profile
    print(f"\nGenerating frontal profile...")
    original_img, generated = tester.generate_frontal(TEST_IMAGE)
    
    # Save result
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(TEST_IMAGE))[0]
    output_path = os.path.join(OUTPUT_DIR, f'{base_name}_result.png')
    
    tester.save_result(original_img, generated, output_path, show_comparison=True)
    
    print(f"\n{'='*60}")
    print("Testing completed!")
    print('='*60)


if __name__ == '__main__':
    main()
