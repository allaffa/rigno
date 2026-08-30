<h1 align="center"> Region Interaction Graph Neural Operator </h1>
<p align="center"> <img src="assets/camlab-logo.png" width="100"/> </p>
<h3 align="center"> <a href="https://arxiv.org/abs/2501.19205"> "RIGNO: A Graph-based framework for robust and accurate operator learning for PDEs on arbitrary domains" </a>  </h3>

<h5 align="center">  Sepehr Mousavi, Shizheng Wen, Levi Lingsch, </h5>
<h5 align="center">  Maximilian Herde, Bogdan Raonic, Siddhartha Mishra</h5>



<h4 align="center">  Abstract </h4>

<p align="center">  Learning the solution operators of PDEs on arbitrary domains is challenging due to the diversity of possible domain shapes, in addition to the often intricate underlying physics. We propose an end-to-end graph neural network (GNN) based neural operator to learn PDE solution operators from data on point clouds in arbitrary domains. Our multi-scale model maps data between input/output point clouds by passing it through a downsampled regional mesh. Many novel elements are also incorporated to ensure resolution invariance and temporal continuity. Our model, termed RIGNO, is tested on a challenging suite of benchmarks, composed of various time-dependent and steady PDEs defined on a diverse set of domains. We demonstrate that RIGNO is significantly more accurate than neural operator baselines and robustly generalizes to unseen grid resolutions and time instances. </p>

<hr>

<!-- ## Architecture -->

<p align="center"> <img src="assets/architecture.png" alt="architecture" width="900"/> </p>

The above figure shows a general schematic of the RIGNO architecture for an idealized two-dimensional domain. The inputs are first independently projected to a latent space by feed-forward blocks. The information on the original discretization (*physical nodes*) is then locally aggregated to a coarser discretization (*regional nodes*). Regional nodes are connected to each other with edges of multiple length scales. Several message passing steps are then applied on the regional nodes which constitute the processor. The processed features are then mapped back to the original discretization by using similar edges as in the encoder, before being independently projected back to the desired output dimension via a feed-forward block without normalization layers.

The following animations illustrate the estimates produced by a RIGNO with 1.9 million parameters trained on 1024 solution trajectories (without fancy pairing strategies) of the incompressible Navier-Stokes equations in a two-dimensional square domain with periodic boundary conditions. All figures corresponds to unstructured versions (random point clouds) of the datasets. The model is trained with snapshots up to time 0.7s; the estimates after this time are considered as extrapolation in time.

<p align="center"> <img src="assets/samples/ns-combined-2.gif"  width="1000" /> </p>

## Datasets

Follow the instructions in [this Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.14765453) for downloading the datasets, and put them in a data directory with the following structure:
```
.../data/
    |__ poseidon/
        |__ ACE.nc
        |__ ...
    |__ unstructured/
        |__ ACE.nc
        |__ AF.nc
        |__ ...
```


## Minimal example

The `example.ipynb` notebook provides a minimal example on how to use the codes. After setting up the environment, you can run it yourself or experiment with RIGNO by changing the parameters.

## Usage

> **PyTorch Geometric port:** The `pytorch-geometric-port` branch currently
> contains the converted RIGNO graph builder, model, and message-passing stack.
> Dataset loading, training, evaluation, time stepping, and legacy Flax checkpoint
> conversion have not yet been ported. The legacy JAX training entry points are
> intentionally unavailable with the PyTorch model until phase two. New PyTorch
> checkpoints are not compatible with Flax checkpoints.

### Optional MG-GNN processor

The PyTorch port includes an optional multigrid processor inspired by the
parallel cross-scale architecture in *MG-GNN: Multigrid Graph Neural Networks
for Learning Multilevel Domain Decomposition Methods*. Unlike RIGNO's default
processor, which combines multiple edge scales in one regional graph, this mode
constructs separate clustered graph levels and exchanges learned messages in
both directions between adjacent levels during every processor step.

The graph builder uses `rmesh_levels` to construct the available hierarchy. The
model selects how many of those levels to consume:

```python
builder = RegionInteractionGraphBuilder(
    periodic=False,
    rmesh_levels=3,
    subsample_factor=2,
    overlap_factor_p2r=1.5,
    overlap_factor_r2p=1.5,
    node_coordinate_freqs=4,
)

model = RIGNO(
    num_outputs=2,
    processor_type="multigrid",
    multigrid_levels=3,
)
```

Coarse coordinates are k-means cluster centers, so they are newly constructed
representatives rather than a subset of fine nodes. Assignment edges cover
every fine node and are retained during edge masking. The default
`processor_type="multiscale"` preserves the original port's behavior.

On Linux, create the virtual environment and install all dependencies with:
```bash
./install_linux.sh
source .venv/bin/activate
```

The installer uses `requirements.txt`, requires Python 3.11 or newer and NumPy
2.x, and
verifies the PyTorch and PyTorch Geometric imports. Set `RIGNO_PYTHON` to use a
specific Python executable or `RIGNO_VENV_DIR` to choose another environment
location:
```bash
RIGNO_PYTHON=python3.11 RIGNO_VENV_DIR=.venv-rigno ./install_linux.sh
```

For a manual installation, create and activate a fresh virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the necessary packages:
```bash
pip install -r requirements.txt
```
> In order to use JAX with GPUs/TPUs, a proper option should be given in `requirements.txt`. Please check [JAX compatibility](https://jax.readthedocs.io/en/latest/installation.html) in order to find the relative option for your hardware. For NVIDIA GPUs, the `[cuda12]` option can be given.

Provided that the dataset is downloaded and placed in `./<dir>/<path>.nc`, you can train a RIGNO on it with the following command:
```bash
python -m rigno.train --datadir '<dir>' --datapath '<path>' --epochs 100 --n_train 512 --n_valid 64
```

For time-independent datasets (`AF` and `Elasticity`), make sure to always pass the `--stepper out` and the `--tau_max 0` flags:
```bash
python -m rigno.train --datadir '<dir>' --datapath 'unstructured/AF' --stepper out --tau_max 0 --epochs 100 --n_train 512 --n_valid 64
```

You can run the following command to see the full list of the command-line arguments of the training module and their default values:
```bash
python -m rigno.train --help
```

When a training is launched, the checkpoints, results, and model configurations will be stored in a specific folder in `./rigno/experiments/`. The path of this folder (`<exp>`) will be printed by the training module. You can check the exact values of the metrics, as well as optimization plots within this folder. It is also possible to provide the training module with an old experiment. Provided that the same configurations of RIGNO are being used, the parameters of the old experiment will be used as initialization of the new training.

Once a training is finished, you can run the test module to assess the performance of a trained model on the test samples:
```bash
python -m rigno.test --exp '<exp>' --datadir '<dir>'
```

The test module infers the model directly (single-step inference with different lead times and input times) and autoregressively (with multiple time marching strategies) and plots the predictions for a few test samples. The most important results will be printed out. Plots and full results can be be found in `./rigno/experiments/<exp>/tests/`. The test module also supports more advanced tests and functionalities which can be enabled via the command-line arguments:
1. testing invariance of the model to different resolutions;
2. testing invariance of the model to different discretizations of the same resolution;
3. testing robustness against noisy inputs; and
4. plotting the statistics of ensemble of estimates with different random seeds.

You can run the following command to see the full list of the command-line arguments of the testing module and their default values:
```bash
python -m rigno.test --help
```

## Citation

```bibtex
@inproceedings{mousavi2025rigno,
  title         = {RIGNO: A Graph-based framework for robust and accurate operator learning for PDEs on arbitrary domains},
  author        = {Sepehr Mousavi and Shizheng Wen and Levi Lingsch and Maximilian Herde and Bogdan Raonić and Siddhartha Mishra},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {38},
  year          = {2025},
}
```
