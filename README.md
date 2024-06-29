# SuperTux Image Classification and Segmentation Project

## Table of Contents
- [SuperTux Image Classification and Segmentation Project](#supertux-image-classification-and-segmentation-project)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Model Architectures](#model-architectures)
    - [CNN for Classification](#cnn-for-classification)
    - [FCN for Segmentation](#fcn-for-segmentation)
  - [Data Preprocessing and Augmentation](#data-preprocessing-and-augmentation)
  - [Training Pipeline](#training-pipeline)
  - [Loss Functions and Optimization](#loss-functions-and-optimization)
  - [Results and Performance](#results-and-performance)
  - [Challenges and Solutions](#challenges-and-solutions)
  - [Technologies and Tools](#technologies-and-tools)

## Project Overview

This project focuses on developing two deep learning models to tackle the challenging SuperTux dataset:
1. A Convolutional Neural Network (CNN) for image classification
2. A Fully Convolutional Network (FCN) for image segmentation

The SuperTux dataset presents unique challenges due to its highly occluded, reflected, and contrasting images, making it an excellent test bed for advanced deep learning techniques.

## Dataset

The SuperTux dataset consists of images from the SuperTux game, featuring:
- Highly occluded scenes
- Reflective surfaces
- High contrast areas
- Significant class imbalance (86% of datapoints belonging to two classes)

## Model Architectures

### CNN for Classification

The CNN model is inspired by ResNet architecture and includes:
- Custom residual blocks
- Skip connections
- Adaptive pooling for flexibility in input sizes

Key features:
- Input normalization within the network
- Dropout layers for regularization
- Batch normalization for stable training


        class CNNClassifier(torch.nn.Module):
            def __init__(self, layers=[16, 32, 64, 128], n_input_channels=3, n_output_channels=6, kernel_size=7):
                super().__init__()
                
                L = []
                c = n_input_channels
                for l in layers:
                    L.append(ResidualBlock(c, l, kernel_size=3))
                    L.append(ResidualBlock(l, l, kernel_size=3))
                    c = l
                
                self.network = torch.nn.Sequential(*L)
                self.classifier = torch.nn.Linear(c, n_output_channels)
                
            def forward(self, x):
                z = self.network(x)
                return self.classifier(z.mean(dim=[2, 3]))


### FCN for Segmentation

The FCN model uses an encoder-decoder structure with:
- Triple convolutional blocks
- Up-convolutions for decoder
- Skip connections between encoder and decoder

Key features:
- Input normalization
- Dropout for regularization


        class FCN(torch.nn.Module):
            def __init__(self, layers=[16, 32, 64, 128], n_input_channels=3, n_output_channels=5):
                super().__init__()
                
                # Encoder
                self.encoder = []
                c = n_input_channels
                for l in layers:
                    self.encoder.append(TripleConv(c, l))
                    c = l
                
                # Decoder
                self.decoder = []
                for l in reversed(layers[:-1]):
                    self.decoder.append(torch.nn.ConvTranspose2d(c, l, kernel_size=3, stride=2, padding=1, output_padding=1))
                    self.decoder.append(TripleConv(l, l))
                    c = l
                
                self.classifier = torch.nn.Conv2d(c, n_output_channels, kernel_size=1)
                
            def forward(self, x):
                z = x
                for encoder_layer in self.encoder:
                    z = encoder_layer(z)
                
                for decoder_layer in self.decoder:
                    z = decoder_layer(z)
                
                return self.classifier(z)


## Data Preprocessing and Augmentation

Extensive data augmentation techniques are employed to improve model generalization:


        transform = dense_transforms.Compose([
            dense_transforms.RandomHorizontalFlip(),
            dense_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            dense_transforms.RandomAffine(degrees=(-15, 15), translate=(0.2, 0.2), scale=(0.8, 1.2)),  
            dense_transforms.ToTensor()
        ])


These transformations include:
- Random horizontal flips
- Color jittering (brightness, contrast, saturation, hue)
- Random affine transformations (rotation, translation, scaling)

## Training Pipeline

A robust training pipeline is implemented with:
- Custom data loaders
- TensorBoard logging for real-time performance tracking
- K-fold cross-validation for robust performance estimation
  

        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

        for epoch in range(args.num_epoch):
            model.train()
            train_acc.reset()
            for img, label in train_data:
                img, label = img.to(device), label.to(device)
                logit = model(img)
                loss_val = loss(logit, label)
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                train_acc.add(logit.argmax(1), label)

            train_logger.add_scalar('accuracy', train_acc.get(), global_step=epoch)


## Loss Functions and Optimization

- Classification: Standard Cross-Entropy Loss
- Segmentation: Weighted Cross-Entropy Loss to address class imbalance


        class_weights = torch.FloatTensor(DENSE_CLASS_DISTRIBUTION).to(device)
        loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)


Optimization:
- Adam optimizer with weight decay
- Learning rate scheduling


        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)


## Results and Performance

- CNN Classification Accuracy: 92%
- FCN Segmentation Score: 78%

Detailed metrics are logged and visualized using TensorBoard, including:
- Global accuracy
- IoU (Intersection over Union) for segmentation
- Learning curves
- Confusion matrices

## Challenges and Solutions

1. Dataset Complexity:
   - Solution: Careful model architecture design and extensive data augmentation

2. Class Imbalance:
   - Solution: Weighted loss function for segmentation task

3. Model Efficiency:
   - Solution: Balanced architecture design, considering both performance and computational requirements

## Technologies and Tools

- PyTorch: Main deep learning framework
- TensorFlow: Used for TensorBoard logging
- NumPy: Numerical computations
- torchvision: Data loading and transformations
- TensorBoard: Performance visualization and logging
- Custom data augmentation pipelines
