# Semi-Supervised Defect Detection with Normalizing Flows

This project applies the Same Same But DifferNet model for semi-supervised defect detection on the BTAD and MVTec datasets. It is based on the WACV 2021 paper "Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows" by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- NVIDIA GPU (We used an NVIDIA L4 GPU with 24GB GPU RAM)
- 16 CPUs or more recommended

### Installation

1. Clone this repository
2. Set up a virtual environment (recommended)
3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The model expects the following directory structure for each dataset:

```
dataset_directory/
    class_01/
        test/
            anomaly/
            good/
        train/
            good/
    class_02/
        ...
```

Prepare your BTAD and MVTec datasets according to this structure.

## Configuration

All configurations can be made in `config.py`. Key parameters include:

- `meta_epochs = 24`
- `sub_epochs = 8`

Modify the main and config files as needed to suit your dataset classes and enable training across all classes.

## Training

To start the training, run:

```bash
python main.py
```

We used the [Lightning AI](https://lightning.ai) platform for GPU resources during training.

## Results

### BTAD Dataset Results

| Class    | Max AUROC | Max AUROC Epoch |
|----------|-----------|-----------------|
| class_01 | 0.9903    | 5               |
| class_02 | 0.8298    | 5               |
| class_03 | 0.9870    | 20              |

### MVTec Dataset Results

| Class      | Last AUROC | Max AUROC | Max AUROC Epoch |
|------------|------------|-----------|-----------------|
| bottle     | 0.9881     | 0.9921    | 1               |
| capsule    | 0.8089     | 0.8432    | 15              |
| grid       | 0.7109     | 0.7937    | 3               |
| leather    | 0.9813     | 0.9823    | 19              |
| pill       | 0.8669     | 0.8792    | 15              |
| tile       | 0.9751     | 0.9978    | 13              |
| transistor | 0.8983     | 0.9142    | 19              |
| zipper     | 0.9249     | 0.9359    | 16              |
| cable      | 0.9496     | 0.9665    | 19              |
| carpet     | 0.9057     | 0.9109    | 21              |
| hazelnut   | 0.9939     | 0.9979    | 19              |
| metal_nut  | 0.9580     | 0.9619    | 15              |
| screw      | 0.9424     | 0.9572    | 17              |
| toothbrush | 0.9722     | 0.9833    | 16              |
| wood       | 0.9930     | 0.9974    | 17              |


## Credits

This project builds upon the work of Marco Rudolph, Bastian Wandt, and Bodo Rosenhahn. Some code from the FrEIA framework was used for the implementation of Normalizing Flows.

## Citation

If you use this work in your research, please cite the original paper:

```bibtex
@inproceedings{RudWan2021,
    author = {Marco Rudolph and Bastian Wandt and Bodo Rosenhahn},
    title = {Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows},
    booktitle = {Winter Conference on Applications of Computer Vision (WACV)},
    year = {2021},
    month = jan
}
```

## License

This project is licensed under the MIT License.
