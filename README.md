# week-5-gans
🖼️ Monet Style Transfer with GANs

CS Project — Kaggle “I’m Something of a Painter Myself” Competition

📌 Overview

This project trains a Generative Adversarial Network (GAN) to transform real photos into Monet-style paintings.
The work is based on the Kaggle competition:
👉 https://www.kaggle.com/competitions/gan-getting-started

The objective is to generate 7,000+ Monet-style images and create a submission file (images.zip) scored with MiFID (Memorization-informed Fréchet Inception Distance). Lower scores indicate more realistic and less memorized outputs.

This repository contains:

A Kaggle-ready Jupyter notebook with EDA, preprocessing, model architecture, training loop, and submission generation.

Supporting code for data pipelines, generator & discriminator models, and output visualization.

A full write-up following the project rubric (problem description, EDA, analysis, architecture, results, and conclusion).

📂 Project Structure
.
├── notebook.ipynb          # Main Kaggle notebook (EDA → model → training → submission)
├── README.md               # Project documentation (you are reading it!)
└── assets/                 # (Optional) Images, samples, plots, screenshots


Note: The dataset itself is not stored here, as Kaggle provides it directly in the notebook environment.

🧠 Problem Description

The goal is to train a model that learns a mapping from real photos → Monet-style images, capturing brush strokes, color palettes, and texture without losing underlying scene structure.

The dataset includes:

~300 Monet paintings

~7,000 landscape photos
All images are resized to 256×256 RGB for training.

The final submission consists of 7,000 generated images zipped into images.zip.

🔍 Exploratory Data Analysis (EDA)

Key steps included:

Verifying dataset structure

Visualizing sample Monet paintings and photos

Checking image sizes / channels

Normalizing and preparing data with tf.data

Setting up input pipelines with batching, shuffling, and prefetching

Insights from EDA directly informed the architecture choice (style-focused editing vs structure generation).

🏗️ Model Architecture

This project uses a GAN-style image-to-image translation model:

Generator (Photo → Monet)

Encoder–Residual–Decoder structure

Residual blocks for style transformation while preserving content

Final tanh layer for outputs in [-1, 1]

Inspired by CycleGAN-style residual architectures

Discriminator (PatchGAN)

Convolutional PatchGAN classifier

Predicts real/fake for image patches rather than whole images

Encourages realistic local texture (brush strokes, Monet color gradients)

Losses

Adversarial loss (Binary Cross-Entropy)

L1 reconstruction loss to maintain photo content

Adam optimizers (2e-4 learning rate, β1 = 0.5)

🏃‍♀️ Training

Training is performed in a Kaggle GPU environment.

Key components:

Custom train_step() using tf.GradientTape

Alternating generator & discriminator updates

Sanity-check images generated each epoch

Loss curves logged for both networks

Batch size was adjusted for Kaggle GPU memory constraints.

📦 Submission

After training, the generator is run over all ~7,000 photos.
Each Monet-style output is saved as .jpg, then zipped into:

images.zip


This file is uploaded to Kaggle for scoring.

📊 Results

(Replace this with your actual results once finalized.)

MiFID Score: X.XX

Generator outputs capture Monet-style color shifts and textures

Residual blocks + PatchGAN architecture produced noticeably better results than simpler models

You can include sample images or loss plots here if you want.

🔚 Conclusion

This project demonstrates how GANs can learn artistic style transfer from unpaired data.
Key learnings:

Residual blocks stabilize style transformation

L1 content loss is crucial for preserving scene structure

PatchGAN discriminators enforce realistic Monet-like texture

GPU memory limits affect batch size on Kaggle

MiFID rewards realistic + non-memorized generations

Future improvements could include:

A full CycleGAN setup (two generators + two discriminators)

Data augmentation for Monet paintings

Learning-rate scheduling

Larger models with gradient checkpointing

🛠️ Technologies Used

Python

TensorFlow / Keras

NumPy / Matplotlib

Kaggle GPU environment

🙌 Acknowledgments

Kaggle for dataset + evaluation

TensorFlow documentation and image-to-image translation examples

Course instructors and reference GAN papers (Goodfellow et al., CycleGAN, Pix2Pix)
