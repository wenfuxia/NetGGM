# Covariance Selection over Networks

This repository contains the implementation of **NetGGM**, a distributed sparse precision matrix estimation algorithm.

## Prerequisites

Before running the code, ensure you have MATLAB installed with the **Parallel Computing Toolbox**. This toolbox is essential for efficient distributed computations.

## Getting Started

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/wenfuxia/NetGGM.git
2. Open MATLAB and navigate to the cloned repository folder.
3. Run the `main.m` script.
4. After running `main.m`, the following variables will be saved:
- `Theta_opt`: Results from the **G-ISTA** algorithm.
- `Theta_bl`: Results from **two baseline methods**.
- `Theta`: Results from **NetGGM** with Metropolis weights.
- `Theta_L`: Results from **NetGGM** with Laplacian weights.
- `Theta_M`: Results from **NetGGM** with Uniform weights.
- `times`: CPU time from **G-ISTA**, **two baseline methods**, and **NetGGM** with Metropolis weights.
- `times_L`: CPU time from **NetGGM** with Laplacian weights.
- `times_M`: CPU time from **NetGGM** with Uniform weights.
