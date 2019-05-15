# Pycptmodels

Pycptmodels is a package containing models for clustered photolithography tools (CPTs) used in semiconductor 
fabrication. It offers an easy way to simulate CPTs with various equipment models (multi-class flowline, exit recursion 
models, affine and linear models). This package can also generate random arrivals of lots according to a Poisson 
Process.

## Installation

- Clone this repository and unzip to desired location.
- Install:

```
pip install -U <package_dir>
```


## Usage
- Import package.
- Create Input object with desired parameters and generate random sample:

```
input1 = PoissonProcessInput(N=20, lambda_=1000, lotsizes=[3, 4, 5, 6, 7],
                                 lotsize_weights=[0.05, 0.2, 0.5, 0.2, 0.05], reticle=[210, 260], prescan=[240, 420],
                                 K=3)
input1.initialize()
```

- Construct a flowline model for the desired CPT configuration. This model will be assumed to be exact. By default, it will use the CPT configuration outlined
in the papers below. Use `initialize()` to apply the flowline model to the CPT configuration. 

```
FL = ParametricFlowLine(
        flow=[
            [1, 1, 1, 1, 1, 2, 2, 3, 4, 3, 3, 3, 3, 2, 2, 1],
            [1, 1, 1, 1, 1, 1, 2, 2, 4, 3, 3, 3, 3, 2, 2, 1],
            [1, 1, 1, 1, 2, 2, 2, 1, 2, 3, 4, 3, 3, 3, 3, 2, 2, 1], ],
        R=[
            [1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 3, 2, 2, 1],
            [1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 3, 2, 2, 1],
            [1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 3, 2, 2, 1], ],
        PT=[
            [0, 80, 90, 60, 65, 50, 90, 60, 100, 90, 60, 90, 130, 90, 60, 0],
            [0, 80, 90, 60, 65, 90, 60, 50, 100, 90, 60, 90, 130, 90, 60, 0],
            [0, 80, 90, 60, 50, 90, 60, 65, 90, 60, 100, 90, 60, 90, 130, 90, 60, 0], ],
        buffer_R=[1, 1, 16],
        move=3,
        pick=1
    )
FL.initialize()
```

- Run the flowline model to calculate lot start and completion times.

```
FL.run(input1)
```

- Train an affine model on the lot start and completion times given by the flowline model.

```
affine = AffineModel()
affine.train(input1, FL.C, FL.C_w)
```

- Run the affine model on an input (the same input used for training is used here) to calculate lot start and completion times estimated by the affine model.

```
affine.run(input1)
affine.S
affine.C
```

For more details, refer to the docstrings included in the code.

## Testing
Tests are located in the `tests` directory.
Use `pytest`.

```
pytest -v
```


### Relevant links:
1. [Models of Clustered Photolithography Tools for Fab-Level Simulation: From Affine to Flow Line](https://ieeexplore.ieee.org/abstract/document/8038810)
2. [Exit Recursion Models of Clustered Photolithography Tools for Fab Level Simulation](https://ieeexplore.ieee.org/abstract/document/7605455)