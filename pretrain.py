import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os
from functools import partial
from tqdm import tqdm
from .datagen import DenoMAEDataGenerator
from .main import DenoMAE2

def parse_args():
    parser = argparse.ArgumentParser(description="DenoMAE2.0 Training Script")
    parser.add_argument("--train_path", type=str, default="/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/unlabeled_10k/train/",
                        help="Path to training data")
    parser.add_argument("--test_path", type=str, default="/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/unlabeled_10k/test/",
                        help="Path to test data")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--image_size", type=int, nargs=2, default=(224, 224), help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the model")
    parser.add_argument("--in_chans", type=int, default=3, help="Number of input channels")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--decoder_embed_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--encoder_depth", type=int, default=12, help="Depth of the encoder")
    parser.add_argument("--decoder_depth", type=int, default=8, help="Depth of the decoder")
    parser.add_argument("--encoder_num_heads", type=int, default=12, help="Number of encoder attention heads")
    parser.add_argument("--decoder_num_heads", type=int, default=8, help="Number of decoder attention heads")
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--log_dir", type=str, default='runs/denoMAE2', help="Directory for TensorBoard logs")
    parser.add_argument("--model_dir", type=str, default='models', help="Directory to save models")
    parser.add_argument("--num_modality", type=int, default=1, help="Number of modalities")
    parser.add_argument("--model_name", type=str, default='denoMAE2_2', help="Name of the model file")
    parser.add_argument("--final_model_name", type=str, default='denoMAE2_final', help="Name of the final model file")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for data loader")
    parser.add_argument("--load", action="store_true", default=False, help="Load a previously saved model")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for AdamW optimizer")
    parser.add_argument("--cls_loss_ratio", type=float, default=0.1, help="Classification loss weight")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Mask ratio for the model")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--test_batch_size", type=int, default=10, help="Batch size for testing")
    return parser.parse_args()

def main():
    args = parse_args()

    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)

    config = {
        'train_noisy_image_path': os.path.join(args.train_path, 'noisyImg/'),
        'train_noiseless_image_path': os.path.join(args.train_path, 'noiseLessImg/'),
        'test_noisy_image_path': os.path.join(args.test_path, 'noisyImg/'),
        'test_noiseless_image_path': os.path.join(args.test_path, 'noiseLessImg/'),
        'batch_size': args.batch_size,
        'test_batch_size': args.test_batch_size,
        'image_size': args.image_size,
        'patch_size': args.patch_size,
    }

    transform = transforms.ToTensor()

    # Create dataset and data loader
    train_dataset = DenoMAEDataGenerator(noisy_image_path=config['train_noisy_image_path'], noiseless_img_path=config['train_noiseless_image_path'], 
                                        image_size=config['image_size'], transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=args.n_workers, pin_memory=True)

    test_dataset = DenoMAEDataGenerator(noisy_image_path=config['test_noisy_image_path'], noiseless_img_path=config['test_noiseless_image_path'], 
                                        image_size=config['image_size'], transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=True, num_workers=args.n_workers, pin_memory=True)

    # TensorBoard setup
    writer = SummaryWriter(args.log_dir)

    # Initialize the model
    model =DenoMAE2(
        img_size=args.image_size[0], 
        patch_size=args.patch_size, 
        embed_dim=args.embed_dim, 
        depth=args.encoder_depth, 
        num_heads=args.encoder_num_heads,
        decoder_embed_dim=args.decoder_embed_dim, 
        decoder_depth=args.decoder_depth, 
        decoder_num_heads=args.decoder_num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )

    # Load model if a checkpoint exists
    if args.load and os.path.exists(args.model_dir+'/'+args.model_name+'.pth'):
        print("Loading model from checkpoint...")
        checkpoint = torch.load(args.model_dir+'/'+args.model_name+'.pth')
        model.load_state_dict(checkpoint)
    else:
        print("No checkpoint found, initializing model.")

    # Check if CUDA is available and set up DataParallel
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)

    # Optimizer setup
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs, 
        eta_min=1e-6
    )

    # Initialize a variable to store the minimum test loss
    best_test_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        # Wrap the training DataLoader with tqdm
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Training")

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):

            inputs, targets = inputs.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            loss_main, pred, mask, loc_loss, reconstructions = model(inputs, targets, mask_ratio=args.mask_ratio)

            # Ensure loss is reduced across GPUs
            if isinstance(loss_main, torch.Tensor) and loss_main.dim() > 0:
                loss_main = loss_main.mean()  # Reduce loss across devices
                loc_loss = loc_loss.mean()
        
            loss = loss_main + loc_loss * args.cls_loss_ratio

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
            # Optimizer step
            optimizer.step()
            
            total_loss += loss.item()

            progress_bar.set_postfix({'train_loss': loss.item()})
            
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Average Loss: {avg_loss:.4f}")
        writer.add_scalar('Average Training Loss', avg_loss, epoch)

        # Learning rate scheduling
        scheduler.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_loss = 0
            # Wrap the evaluation DataLoader with tqdm
            progress_bar = tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Evaluation")
            for batch_idx, (inputs, targets) in enumerate(test_dataloader):

                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass with random masking
                loss_main, pred, mask, loc_loss, reconstructions = model(inputs, targets, mask_ratio=args.mask_ratio)

                # Ensure loss is reduced across GPUs
                if isinstance(loss_main, torch.Tensor) and loss_main.dim() > 0:
                    loss_main = loss_main.mean()  # Reduce loss across devices
                    loc_loss = loc_loss.mean()
            
                loss = loss_main + loc_loss * args.cls_loss_ratio
                
                # Combine losses
                test_loss += loss.item()

                progress_bar.set_postfix({'test_loss': loss.item()})

                # TensorBoard logging for visualization
                grid_data = []
                for i in range(inputs.size(0)):
                    grid_data.extend([
                        (inputs[i] - inputs[i].min()) / (inputs[i].max() - inputs[i].min() + 1e-5),
                        (reconstructions[i] - reconstructions[i].min()) / (reconstructions[i].max() - reconstructions[i].min() + 1e-5),
                        (targets[i] - targets[i].min()) / (targets[i].max() - targets[i].min() + 1e-5),
                    ])

                grid = make_grid(grid_data, nrow=3, normalize=False)
                writer.add_image(f"Epoch {epoch+1}, Batch {batch_idx}", grid, epoch * len(test_dataloader) + batch_idx)

            avg_test_loss = test_loss / len(test_dataloader)
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Test Loss: {avg_test_loss:.4f}")
            writer.add_scalar('Test Loss', avg_test_loss, epoch)

            # Save the model if the current test loss is lower than the best one seen so far
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.module.state_dict(), os.path.join(args.model_dir, args.model_name+'.pth'))
                print(f"New best model saved at epoch {epoch+1} with test loss: {best_test_loss:.4f}")

    # Save the final model with a unique name
    final_model_path = os.path.join(args.model_dir, args.final_model_name + '.pth')
    torch.save(model.module.state_dict(), final_model_path)
    print(f"Final model saved at: {final_model_path}")

    print("Training completed!")

if __name__ == "__main__":
    main()   
