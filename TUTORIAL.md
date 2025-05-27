# AA-CLIP: A Comprehensive Tutorial

This tutorial provides a comprehensive overview of the AA-CLIP repository, explaining its purpose, code structure, training and inference processes, key components, and how to get started with running the code.

## 1. Introduction to AA-CLIP

**Project Goal:**
AA-CLIP (Anomaly-Aware CLIP) aims to significantly enhance the capabilities of CLIP (Contrastive Language-Image Pre-training) for zero-shot anomaly detection tasks.

**Core Idea:**
The fundamental concept is to make the standard CLIP model "anomaly-aware." While CLIP excels at general visual and text understanding, it isn't inherently optimized for identifying anomalies or out-of-distribution data. AA-CLIP addresses this by adapting CLIP to better recognize what constitutes an anomaly, thereby improving its discrimination between normal and abnormal features.

**Method:**
This is achieved through a two-stage adaptation process that refines both the text and visual feature extraction capabilities of CLIP:
1.  **Text Feature Adaptation:** The text encoder of CLIP is adapted to create more discriminative textual descriptions (or "prompts") that can effectively distinguish between normal and anomalous data. This involves fine-tuning parts of the text encoder to make these prompts more sensitive to anomaly-related semantics.
2.  **Visual Feature Adaptation:** Subsequently, the visual encoder of CLIP is adapted to better align visual features with these newly refined, anomaly-aware text features. This ensures that the visual representations become more sensitive to the subtle (or overt) visual cues that differentiate normal instances from anomalies, as guided by the adapted text prompts.

By specializing both modalities, AA-CLIP can identify anomalies in a zero-shot manner, meaning without prior training on specific anomaly classes for the target dataset.

## 2. Repository Code Structure

The AA-CLIP project is organized with the following key files and directories:

-   **`train.py`**:
    *   The main script for training the AA-CLIP model.
    *   It handles dataset loading, model initialization (including the `AdaptedCLIP` model and its adapters), optimizer setup, loss function definition, and the execution of the two-stage training loop.

-   **`test.py`**:
    *   The main script for evaluating a trained AA-CLIP model and performing inference on test datasets.
    *   It loads trained model weights, processes test data, calculates anomaly scores/maps, and computes various evaluation metrics.

-   **`model/`**: This directory contains all core model components:
    *   **`model/adapter.py`**: Defines the crucial `AdaptedCLIP` class. This class wraps the standard CLIP model and integrates the learnable text and image adapter modules. It's where the logic for the two-stage adaptation is implemented.
    *   **`model/clip.py`**: Likely contains the implementation or interface for the base CLIP model architecture upon which `AdaptedCLIP` is built.
    *   Other files like `model.py`, `adapter_modules.py`, `modified_resnet.py`, and `transformer.py` provide building blocks such as the adapter modules (`SimpleAdapter`, `SimpleProj`) and components of the vision and language models.

-   **`dataset/`**: This directory manages all data-related operations:
    *   **`dataset/__init__.py`**: Contains the `get_dataset` function, which is the primary interface for loading training and testing datasets. It also includes `BaseDataset` and `BaseSingleClassDataset` classes that handle data loading, preprocessing, and transformations.
    *   **`dataset/constants.py`**: Stores constants like dataset paths, class names, and prompt templates used throughout the project.
    *   **`dataset/metadata/`**: Contains `.jsonl` files that provide metadata (image paths, labels, mask paths, class names) for various datasets.

-   **`forward_utils.py`**:
    *   A collection of essential utility functions used during the model's forward pass, loss calculation, and text embedding generation.
    *   Includes functions like `get_adapted_text_embedding` (for creating anomaly-aware text prompts) and `calculate_similarity_map` (for generating anomaly heatmaps by comparing image and text features).

-   **`README.md`**:
    *   The main documentation file. Provides an overview of the project, installation instructions, download links for datasets and pre-trained models, and basic commands for running training and evaluation.

-   **`requirements.txt`**: Lists the Python dependencies required for the project.
-   **`scripts.sh`**: An optional shell script that may automate training and testing across multiple datasets.

## 3. Training Process (`train.py`)

The training process in `train.py` is orchestrated by the `main()` function and is divided into two sequential stages, focusing on adapting the `AdaptedCLIP` model:

**A. Initialization and Setup:**

1.  **`AdaptedCLIP` Model Instantiation:** An `AdaptedCLIP` model (defined in `model/adapter.py`) is created. This model wraps a standard pre-trained CLIP model (e.g., ViT-L-14-336) and includes `text_adapter` and `image_adapter` modules which are initially learnable.
2.  **`clip_surgery` Model:** A separate instance of the CLIP model, termed `clip_surgery`, is created. Its visual encoder might undergo modifications (e.g., `DAPM_replace`). This model is used with frozen weights during the text adapter training stage to provide stable image features.
3.  **Optimizers:** Two Adam optimizers are configured:
    *   `text_optimizer`: For training the parameters of `model.text_adapter`.
    *   `image_optimizer`: For training the parameters of `model.image_adapter`.
4.  **Schedulers:** A `MultiStepLR` learning rate scheduler (`image_scheduler`) is typically used for the `image_optimizer`.
5.  **Datasets:** The `get_dataset` function (from `dataset/__init__.py`) is called to load the training data. It usually returns two versions: `text_dataset` (potentially with specific augmentations for text adapter training) and `image_dataset`.
6.  **Checkpointing:** The script supports resuming training by loading states for the model adapters and optimizers from existing checkpoint files (e.g., `text_adapter.pth`, `image_adapter.pth` located in the save path).

**B. Stage 1: Text Adapter Training (`train_text_adapter` function):**

1.  **Objective:** To refine the text embeddings (prompts like "normal" and "abnormal") to be more discriminative for anomalies. The `text_adapter` within the `AdaptedCLIP` model is trained in this stage.
2.  **Process:**
    *   The function iterates for a specified number of `text_epoch`s.
    *   **Adapted Text Embeddings:** For each batch, `get_adapted_single_class_text_embedding` (from `forward_utils.py`) generates text features. These embeddings are produced by the `AdaptedCLIP` model itself, meaning the `text_adapter` actively influences them during its own training.
    *   **Frozen Image Features:** The `clip_surgery` model (with `torch.no_grad()`) encodes the input images, providing image patch features. These features are from a fixed, pre-trained encoder and serve as stable targets for the text adaptation.
    *   **Loss Calculation:**
        *   *Segmentation Loss*: `calculate_similarity_map` (from `forward_utils.py`) compares the frozen image patch features with the currently adapted text features to produce an anomaly map. A segmentation loss (e.g., Dice + Focal Loss, defined in `forward_utils.py::calculate_seg_loss`) is computed against ground truth masks.
        *   *Orthogonal Loss*: An additional loss term often encourages orthogonality between "normal" and "abnormal" text embeddings, promoting their distinctiveness.
    *   **Optimization:** The total loss is backpropagated, and the `text_optimizer` updates the weights of `model.text_adapter`.
3.  **Checkpointing:** The state of the `text_adapter` and its optimizer is saved after each epoch to a file like `text_adapter.pth` in the specified save path.

**C. Intermediate Step: Cache Adapted Text Embeddings:**

1.  After the text adapter training is complete (or if skipped), the `get_adapted_text_embedding` function (from `forward_utils.py`) is called.
2.  This function uses the `AdaptedCLIP` model (with its newly trained `text_adapter`) to compute and cache the final adapted text embeddings for all classes in the dataset. These fixed text embeddings will serve as targets for the next stage.
3.  The `clip_surgery` model and text-related components are often deleted to free memory.

**D. Stage 2: Image Adapter Training (`train_image_adapter` function):**

1.  **Objective:** To adapt the visual features extracted by CLIP to align better with the (now fixed) anomaly-aware text embeddings produced in Stage 1. The `image_adapter` within `AdaptedCLIP` is trained here.
2.  **Process:**
    *   The function iterates for a specified number of `image_epoch`s.
    *   **Fixed Text Embeddings:** The cached, adapted text embeddings from the previous step are used for the current batch.
    *   **Adapted Image Features:** Input images are passed through the `AdaptedCLIP` model. This time, the `image_adapter` (which includes `layer_adapters` and projection layers) processes the visual features, yielding `patch_features` (for segmentation) and a global `det_feature` (for image-level classification/detection).
    *   **Loss Calculation:**
        *   *Classification Loss*: The global `det_feature` is compared with the fixed adapted text embeddings to get image-level classification predictions (e.g., normal/abnormal). A cross-entropy loss is typically computed.
        *   *Segmentation Loss*: The adapted `patch_features` are compared with the fixed adapted text embeddings using `calculate_similarity_map` (from `forward_utils.py`), and a segmentation loss (via `calculate_seg_loss` from `forward_utils.py`) is computed against ground truth masks.
    *   **Optimization:** The combined loss is backpropagated, and the `image_optimizer` updates the weights of `model.image_adapter`. The `image_scheduler` adjusts the learning rate.
3.  **Checkpointing:** The state of the `image_adapter` and its optimizer is saved regularly (e.g., to `image_adapter.pth` and epoch-specific files like `image_adapter_{epoch_number}.pth` in the specified save path).

This two-stage approach systematically makes CLIP "anomaly-aware" by first specializing its textual understanding of anomalies and then tuning its visual feature extraction to match these specialized textual cues.

## 4. Inference Process (`test.py`)

The `test.py` script evaluates the performance of a trained `AdaptedCLIP` model. The `main()` function manages loading the model and data, while `get_predictions` handles the core inference logic.

**A. Setup in `main()` (before `get_predictions`):**

1.  **Model Loading:**
    *   An `AdaptedCLIP` model (from `model/adapter.py`) is instantiated similarly to training.
    *   The trained `text_adapter` state is loaded from its checkpoint (e.g., `text_adapter.pth` from the save path).
    *   The script typically iterates through multiple saved checkpoints of the `image_adapter` (e.g., `image_adapter_{epoch_number}.pth` from the save path) to evaluate performance at different training stages.
    *   The model is set to evaluation mode (`model.eval()`).
2.  **Adapted Text Embeddings:** `get_adapted_text_embedding` (from `forward_utils.py`) is called to compute and load the anomaly-aware text embeddings for all classes in the target test dataset, using the loaded `text_adapter`.
3.  **Test Data:** `get_dataset` (from `dataset/__init__.py`) loads the test data, usually providing a dictionary of class-specific datasets. A `DataLoader` is created for each class to be evaluated.

**B. Core Inference (`get_predictions` function):**

This function is typically called for each class within the test dataset.

1.  **Inputs:** The `AdaptedCLIP` model (with loaded adapters), `class_text_embeddings` (the specific adapted text features for the current class), and the `test_loader` for that class.
2.  **Iterating through Data:** The function processes images batch by batch from the `test_loader`.
3.  **Image Forward Pass:** For each image:
    *   `patch_features, det_feature = model(image)`: The `AdaptedCLIP` model extracts adapted visual features.
        *   `det_feature`: A global, image-level feature.
        *   `patch_features`: A list of local, patch-based features from different stages of the visual encoder.
4.  **Image-Level Anomaly Score:**
    *   The `det_feature` is compared with the `class_text_embeddings` (e.g., via dot product) to get a score indicating similarity to "normal" and "anomaly" prompts.
    *   This is often converted into a single anomaly score for the entire image (e.g., `(score_anomaly + 1 - score_normal) / 2`).
5.  **Pixel-Level Anomaly Map:**
    *   Each set of features in `patch_features` is compared with the `class_text_embeddings` using `calculate_similarity_map` (from `forward_utils.py`). This function:
        *   Computes similarity scores for each patch.
        *   Combines scores for "normal" and "anomaly" (e.g., `(score_anomaly_patch + 1 - score_normal_patch) / 2`).
        *   Applies Gaussian smoothing to the resulting map.
        *   Upsamples the map to the original image resolution.
    *   Anomaly maps from different feature levels might be aggregated (e.g., summed or averaged).
6.  **Output:** The function returns aggregated lists of ground truth masks, image-level labels, predicted pixel-level anomaly maps, and predicted image-level anomaly scores.

**C. Evaluation in `main()` (after `get_predictions`):**

1.  The outputs from `get_predictions` are passed to `metrics_eval` (in `forward_utils.py`).
2.  This function calculates standard anomaly detection metrics such as:
    *   Pixel-level Area Under the ROC Curve (AUC) and Average Precision (AP).
    *   Image-level AUC and AP.
3.  Results are logged and often stored in a structured format (e.g., Pandas DataFrame).
4.  Optional visualization of anomaly maps on images can be performed using `visualize` (from `forward_utils.py`).

## 5. Key Helper Files and Functions

Several files and functions play crucial roles in the AA-CLIP framework:

-   **`model/adapter.py::AdaptedCLIP`**:
    *   **Role:** The core neural network model. It wraps a standard CLIP model and injects learnable `SimpleAdapter` and `SimpleProj` modules into both the visual and text encoder pathways.
    *   **Functionality:** Manages the forward passes for both image and text data, applying the adapter transformations at specified layers and combining their outputs with the original CLIP features using learned or fixed weights. It produces adapted visual features (for segmentation and detection) and adapted text embeddings.

-   **`forward_utils.py::get_adapted_text_embedding`** (and its helper `get_adapted_single_class_text_embedding`):
    *   **Role:** Generates the crucial "anomaly-aware" text embeddings.
    *   **Functionality:** Takes a class name, formats it with various predefined "normal" and "abnormal" prompt templates (from `dataset/constants.py`), tokenizes these prompts, and then feeds them through the `AdaptedCLIP.encode_text()` method (which uses the trained `text_adapter`). The resulting embeddings for normal and abnormal states are often averaged and normalized, then returned as a pair for each class.

-   **`forward_utils.py::calculate_similarity_map`**:
    *   **Role:** Computes the dense anomaly map at the pixel or patch level.
    *   **Functionality:** Takes image patch features (from `AdaptedCLIP.forward()`) and the adapted text embeddings (normal/abnormal pair from `get_adapted_text_embedding`). It calculates their similarity (e.g., dot product), reshapes this into a map, optionally combines the normal/abnormal scores into a single anomaly score per patch, applies Gaussian smoothing (especially at test time), and upsamples the map to the original image resolution.

-   **`dataset/__init__.py::get_dataset`**:
    *   **Role:** The main factory function for creating and providing `Dataset` objects for training and testing.
    *   **Functionality:** Based on parameters like dataset name, image size, training mode (few-shot/full-shot), and stage (train/test), it locates the appropriate metadata (`.jsonl` files in `dataset/metadata/`) and instantiates either `BaseDataset` (for training, providing data for both text and image adapter stages) or `BaseSingleClassDataset` (for testing, providing data for one specific class at a time). These dataset classes handle image/mask loading from paths specified in metadata and apply necessary transformations.

## 6. How to Run AA-CLIP

Here’s a summary of how to set up and run the AA-CLIP project, based on typical instructions found in the `README.md`:

**A. Prerequisites:**

1.  **Datasets:**
    *   Download the required anomaly detection datasets (e.g., MVTec-AD, VisA, MPDD, various medical imaging datasets). The `README.md` usually provides sources.
    *   Organize these datasets, typically under a common `./data/` directory.
2.  **Metadata (`.jsonl` files):**
    *   Ensure you have the corresponding `.jsonl` files in the `./dataset/metadata/` directory for each dataset you intend to use. These files map image files to their labels, mask paths (if applicable), and class names. The `README.md` details the expected format if you need to create them for custom datasets.
3.  **Pre-trained CLIP Weights:**
    *   Download the official OpenCLIP ViT-L-14-336px model weights.
    *   Place these pre-trained weights into the `./model/` directory.

**B. Installation:**

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/Mwxinnn/AA-CLIP.git
    cd AA-CLIP
    ```
2.  **Set up Conda Environment (Recommended):**
    ```bash
    conda create -n aaclip python=3.10 -y
    conda activate aaclip
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

**C. Training (`train.py`):**

1.  **Command Structure (as per `README.md`):**
    ```bash
    python train.py --shot $shot --save_path $save_path
    ```
    *   `$shot`: This is a placeholder for the number of samples per category for few-shot training (e.g., `8`, `16`, `32`). You would replace `$shot` with the actual number. For full-shot training, this parameter's specific value or handling would be defined by the script's argument parser (often, not setting it or using a value like `-1` implies full shot, but refer to `python train.py --help`).
    *   `$save_path`: This is a placeholder for the directory where training logs and model checkpoints will be saved (e.g., `ckpt/my_experiment`). Replace `$save_path` with your desired path.
    *   **Note:** The `train.py` script itself includes other arguments like `--dataset` (to specify which dataset configuration to use for training, affecting which data is loaded and potentially how prompts are generated via `dataset/constants.py`). For specific training configurations, especially if not using a default dataset, consult `python train.py --help`.

**D. Evaluation/Testing (`test.py`):**

1.  **Command Structure (as per `README.md`):**
    ```bash
    python test.py --save_path $save_path --dataset $dataset
    ```
    *   `$save_path`: This must be the *same path* used during the training phase (placeholder for your actual path), as the script needs to load the saved model adapters from here.
    *   `$dataset`: Placeholder for the name of the dataset you wish to evaluate the trained model on (e.g., `MVTec`, `VisA`). Replace `$dataset` with the actual dataset name.

**E. Using `scripts.sh` (Optional):**

*   The repository may include a `scripts.sh` file. This shell script often contains commands to automate the training and evaluation process across multiple datasets or configurations, serving as a convenient way to reproduce published results.
    ```bash
    bash scripts.sh
    ```

Always refer to the `README.md` for the most current and specific commands and configurations, as argument names or dataset handling specifics can vary.
