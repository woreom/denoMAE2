import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from functools import partial
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
from .main import DenoMAE2 #custom modules always load last


class DownstreamClassifier(nn.Module):
    def __init__(self, denoMAE_model, num_classes, hidden_dim=256, freeze_encoder=True):
        super().__init__()
        self.encoder = denoMAE_model
        self.hidden_dim = hidden_dim
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classification_head = nn.Sequential(
            nn.Linear(denoMAE_model.pos_embed.shape[-1], 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features, _, _ = self.encoder.forward_encoder(x, mask_ratio=0)
        cls_token = features[:, 1:, :].mean(dim=1)
        logits = self.classification_head(cls_token)
        return logits

def train_classifier(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    progress_bar = tqdm(train_loader, desc="Training Classifier")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        progress_bar.set_postfix({'loss': loss.item(), 'accuracy': 100. * correct / total})
    return total_loss / len(train_loader), 100. * correct / total

def evaluate_classifier(model, test_loader, criterion, device, create_confusion_matrix=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating Classifier")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if create_confusion_matrix:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            progress_bar.set_postfix({'loss': loss.item(), 'accuracy': 100. * correct / total})
    
    if create_confusion_matrix:
        return total_loss / len(test_loader), 100. * correct / total, all_preds, all_labels
    return total_loss / len(test_loader), 100. * correct / total

def plot_confusion_matrix(true_labels, pred_labels, class_names, save_dir='experiments/confusion_matrix/'):
    """
    Create and save a confusion matrix visualization.
    
    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        class_names: List of class names
        save_dir: Directory to save the confusion matrix plot
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save the raw confusion matrix
    np.save(os.path.join(save_dir, 'confusion_matrix.npy'), cm)
    print(f"Confusion matrix saved to {save_dir}")

def main(args):
    if torch.cuda.is_available():
        gpus = [int(id) for id in args.gpu.split(',')]
        device = torch.device(f'cuda:{gpus[0]}')  # Primary GPU
        print(f"Using device(s): {gpus}")
    else:
        device = torch.device('cpu')
        gpus = []
        print("Using CPU")

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=args.train_data_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=args.test_data_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    base_encoder = DenoMAE2(
        img_size=args.image_size[0], 
        patch_size=args.patch_size, 
        embed_dim=args.embed_dim, 
        depth=args.encoder_depth, 
        num_heads=args.encoder_num_heads,
        decoder_embed_dim=args.decoder_embed_dim, 
        decoder_depth=args.decoder_depth, 
        decoder_num_heads=args.decoder_num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ).to(device)

    if args.load:
        print(f"Loading pretrained model from {args.pretrained_model_path}")
        checkpoint = torch.load(args.pretrained_model_path, map_location=device)
        base_encoder.load_state_dict(checkpoint)
    else:
        print("Initializing model with random weights")

    model = DownstreamClassifier(base_encoder, args.num_classes).to(device)
    
    # Enable multi-GPU if available
    if torch.cuda.is_available() and len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Add learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs,
        eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss()

    # Get class names from the dataset
    class_names = train_dataset.classes
    print("Class names:", class_names)

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        train_loss, train_acc = train_classifier(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        if (epoch + 1) % 5 == 0:
            test_loss, test_acc = evaluate_classifier(model, test_loader, criterion, device)
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        scheduler.step()

    # Final evaluation with confusion matrix
    print("Performing final evaluation with confusion matrix...")
    test_loss, test_acc, all_preds, all_labels = evaluate_classifier(
        model, test_loader, criterion, device, create_confusion_matrix=True
    )
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names)
    
    # Save model
    torch.save(model.state_dict(), args.output_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a downstream classifier using a pre-trained DenoMAE2 model")
    parser.add_argument('--train_data_path', type=str,
                        default='/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/labeled/0.0_dB/train/', help="Path to the training data")
    parser.add_argument('--test_data_path', type=str,
                        default='/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/labeled/0.0_dB/test/', help="Path to the testing data")
    parser.add_argument("--image_size", type=int, nargs=2, default=(224, 224), help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the model")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--decoder_embed_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--encoder_depth", type=int, default=12, help="Depth of the encoder")
    parser.add_argument("--decoder_depth", type=int, default=8, help="Depth of the decoder")
    parser.add_argument("--encoder_num_heads", type=int, default=12, help="Number of encoder attention heads")
    parser.add_argument("--decoder_num_heads", type=int, default=16, help="Number of decoder attention heads")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and testing")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of classes for classification")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="Weight decay")
    parser.add_argument('--pretrained_model_path', type=str,  default='models/DenoMAE2_final.pth', help="Path to the pretrained DenoMAE2 model")
    parser.add_argument('--output_model_path', type=str, default='models/finetunedClassifier.pth', help="Path to save the fine-tuned model")
    parser.add_argument("--load", action="store_true", default=True, help="Load a previously saved model")
    parser.add_argument('--gpu', type=str, default='0',
                      help='GPU indices to use (comma-separated, e.g., "0,1")')
    parser.add_argument('--create_confusion_matrix', action='store_true', default=True,
                      help='Create and save confusion matrix after training')
    
    args = parser.parse_args()
    main(args)
