"""
Evaluate kNN classification
"""
import argparse
import os, sys
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

from soap import SOAP, WelfordChanEstimator
from get_models import *
from main_salient import none_or_float, none_or_int
from torch import Tensor


def get_cls_model(model:str):
    if model == 'dino_base':
        return get_dino_base()[0]
    elif model == 'dinov2_base':
        return get_dinov2_base()[0]
    elif model == 'mae_base':
        return get_mae_base()[0]
    elif model == 'capi_large':
        return get_capi_large()[0]
    elif model == 'deit3_base':
        return get_deit3_base()[0]
    else:
        raise NotImplementedError(f'Backbone {model} not implemented for knn')


class ValidationAugmentation:
    def __init__(
            self,
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        ):
        self.normalize = transforms.Compose([
        transforms.Resize(256, interpolation=3),  # Resize shortest side to 224px
        transforms.CenterCrop(224),  # Take the center 224x224 region
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    def __call__(self, image):
        if isinstance(image, list):
            # Image is a list of images, make augmentations of each image
            return [self.normalize(x) for x in image]
        else:
            return self.normalize(image)

def load_data(dataset, data_path, batch_size, 
              mean=[0.485, 0.456, 0.406], 
              std=[0.229, 0.224, 0.225], 
              override_extensions = ['.jpg', '.cls'], 
              return_datasets=False,
              num_workers=4):
    aug = ValidationAugmentation(mean, std)
    print(aug.normalize)

    # Load your dataset (args dataset and data_path)
    # Remember to apply the augmentation (aug) to both train and val
    train_dataset = ... # Load your train dataset (e.g. ImageNet)
    val_dataset = ... # Load your val dataset

    train_sampler = SequentialSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        sampler = train_sampler,
        num_workers = num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        sampler = val_sampler,
        num_workers = num_workers,
    )

    if return_datasets:
        return train_loader, val_loader, train_dataset, val_dataset
    else:
        return train_loader, val_loader

def extract_features(encoder, dataloader, project, avg_local_embs:bool, device, vit_feat='out'):
    """Extracts features from the dataset using the encoder."""
    encoder.eval()
    all_embeddings, all_labels = [], []
    i = 0
    len_loader = len(dataloader)
    print('Extracting features...')
    with torch.no_grad():
        print('torch.no_grad()')
        for images, labels in dataloader:
            i += 1
            print(f"Processing batch {i}/{len_loader}")
            images = images.to(device)
            if avg_local_embs:
                embeddings = encoder(images, vit_feat=vit_feat)  # Get local patch embeddings      
            else: embeddings = encoder(images)                   # Get global cls embeddings
            if project is not None: 
                embeddings = project(embeddings)
            if avg_local_embs: embeddings = embeddings.mean(-1) # Average over the patches
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            assert embeddings.size(0) == labels.size(0), f'Size mismatch: embeddings={embeddings.shape}, labels={labels.shape}'
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    print("train_features:", train_features.shape)
    print("test_features:", test_features.shape)
    print("train_labels", train_labels.shape)
    print("test_labels", test_labels.shape)
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = max(1, num_test_images // num_chunks)
    print("Number of test images:", num_test_images)
    print("Number of images per chunk:", imgs_per_chunk)
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5

def main():
    parser = argparse.ArgumentParser("Evaluate kNN classification")
    parser.add_argument('--dataset', type=str, default='IN1k')
    parser.add_argument('--datafolder', type=str)
    parser.add_argument('--mean', nargs='+', type=float, default= [0.485, 0.456, 0.406], help='Data normalization mean')
    parser.add_argument('--std', nargs='+', type=float, default=[0.229, 0.224, 0.225], help='Data normalization std')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for dataloader')
    parser.add_argument('--backbone', type=str, default='dinov2_base')
    parser.add_argument('--save_path', type=str, default='kNN_cls_results/results.json')
    parser.add_argument('--checkpoint_folder', type=str, default='weights')
    parser.add_argument('--suppress_pca_components', type=int, nargs='+', default=None)
    parser.add_argument('--semantic_invariance_projector', action='store_true', default=False)
    parser.add_argument('--k', type=int, default=20, help='Number of neighbors for kNN')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature scaling factor for kNN')
    parser.add_argument('--avg_local_embs', action='store_true', default=False)
    parser.add_argument('--features', type=str, default='out')
    parser.add_argument('--score_version', type=str, choices=['scores', 'scaled'],  default='scaled')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--mu', type=none_or_float, default=2.0)
    parser.add_argument('--tau', type=none_or_float, default=0.05)
    parser.add_argument('--comment', type=str, default=None)
    
    args = parser.parse_args()

    # Validate args
    if args.semantic_invariance_projector:
        msg = f'Cannot do manual truncation with --suppress_pca_components and use --semantic_invariance_projector at the same time.'
        assert args.suppress_pca_components == None, msg
    if not args.avg_local_embs:
        msg = f'Cannot evaluate on {args.features} features with cls token. Use --avg_local_embs or --features out'
        assert args.features == 'out', msg
    if args.score_version != 'scaled': 
        args.mu = None
        args.tau = None
    
    # Print args
    for k, v in vars(args).items(): print(f'{k}: {v}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure save dir exists
    directory = os.path.dirname(args.save_path)
    os.makedirs(directory, exist_ok=True)

    # Get backbone
    if args.avg_local_embs:
        encoder = get_dense_backbone(args.backbone).to(device).eval()
    else: 
        encoder = get_cls_model(args.backbone).to(device).eval()

    # Freeze parameters for inference  
    for param in encoder.parameters():
        param.requires_grad = False

    # Get projector
    if args.suppress_pca_components is not None:
        wce = WelfordChanEstimator.deserialize(os.path.join(args.checkpoint_folder, f"{args.backbone}_cov.pth")).to(device)
        project = SOAP.manual_truncation(wce, args.suppress_pca_components)
    elif args.semantic_invariance_projector:
        project = SOAP.from_modelname(
            args.backbone, args.checkpoint_folder, alpha=args.gamma, mu=args.mu, tau=args.tau,
            score_version=args.score_version
        ).to(device)
    else:
        project = None

    if isinstance(project, SOAP):
        if args.avg_local_embs:
            def forward(self, x):
                return (x @ self.projector.mT).mT
            project.forward = MethodType(forward, project)
        else: # Double check these...
            def forward(self, x):
                return (x @ self.projector.mT)
            project.forward = MethodType(forward, project)


    # Load train and val data
    train_loader, test_loader = load_data(args.dataset, args.datafolder, 
        batch_size=args.batch_size, mean=args.mean, std=args.std)

    # Extract features
    train_features, train_labels = extract_features(encoder, train_loader, project, args.avg_local_embs, device, args.features)
    test_features, test_labels = extract_features(encoder, test_loader, project, args.avg_local_embs, device, args.features)
    print("Number of train samples:", train_features.shape[0])
    print("Number of test samples:", test_features.shape[0])

    # Convert to torch tensors
    train_features = torch.tensor(train_features).to(device)
    train_labels = torch.tensor(train_labels).to(device)
    test_features = torch.tensor(test_features).to(device)
    test_labels = torch.tensor(test_labels).to(device)

    # Normalize
    train_features = torch.nn.functional.normalize(torch.tensor(train_features), dim=1, p=2)
    test_features = torch.nn.functional.normalize(torch.tensor(test_features), dim=1, p=2)

    # kNN eval
    top1, top5 = knn_classifier(train_features, train_labels, test_features, test_labels, args.k, args.temperature)
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")

    # Append results
    result = {
        'model': args.backbone,
        'dataset': args.dataset,
        'semantic_invariance_projector' : args.semantic_invariance_projector,
        'checkpoint_folder' : args.checkpoint_folder,
        'suppress_pca_components' : args.suppress_pca_components,
        'avg_local_embs' : args.avg_local_embs,
        'comment' : args.comment,
        'k': args.k,
        'top1_accuracy': top1,
        'top5_accuracy': top5
    }
    
    # Append to the JSON file
    if os.path.exists(args.save_path):
        with open(args.save_path, 'r') as f:
            data = json.load(f)
    else:
        data = []

    data.append(result)

    with open(args.save_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()
