# Setup

```bash
git clone git@github.com:ram1123/fitting.git
cd fitting
```

# Envrionment & Run

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106b_cuda/x86_64-el8-gcc11-opt/setup.sh
python fit.py
```

# Some information

There are three files:

- `fit.py`: The main file that we need to run.
- `plot_config.yaml`: The configuration file for the plots. Where the path of the input file, and what kind of fit we want to do. For example, binned or unbinned.
- `DistributionCompare.py`: This is the class that makes plots or fit.
    - Member function `fit_dimuonInvariantMass_DCBXBW` is for the binned fit.
    - Member function `fit_dimuonInvariantMass_DCBXBW_Unbinned` is for the unbinned fit.
        - For unbinned fit if the cuda is available it will automatically use the cuda to fit. I did it as follows:

        ```python
        if rt.RooFit.EvalBackend.Cuda() is not None:
            print("CUDA available, using GPU")
            fit_result = model.fitTo(data, rt.RooFit.EvalBackend.Cuda(), Save=True, SumW2Error=True) # use GPU
        else:
            print("CUDA not available, using CPU")
            fit_result = model.fitTo(data, EvalBackend="cpu", Save=True, SumW2Error=True)
        ```
