# agriculture-embeddings

This repository implements Quantum Embedding from "Quantum embedding of knowledge for reasoning.", a paper published in NerurlISP 2019 on RiceDO and TreatO. The code in this repo is based on the original implementation of the paper's authors [repo](https://github.com/IBM/e2r/) and pykeen implementation for RiceDO and TreatO.

## Instructions to use this repo

1. Download both datasets `RiceDO` and `TreatO`. Then, put them in `ricedo/data`, `treato/data` and rename them `RiceDO-Version2.owl` and `TreatO-Version2.owl` respectively.

2. Install python environment. We tested on only python 3.12.3 on Windows 11, if you want to use other version, do it on your own risk. You can install every using `conda`. The code will use CUDA automatically if you have torch with cuda install. However, we only tested on CPU only.

    ```bash
    conda env create -f env.yml
    conda activate e2r
    ```

    There might be chances that some package won't install on other OS (such as windows API) or worse the packages didn't register in conda registry. You can try install python 3.12.3 and install packages on your own using pip. 
    Note 16/7/2024: pykeen 1.10.2 cannot run with numpy 2.0.0 (its pretty new at the moment). So, i add specification for numpy (==1.26.4).

    ```bash
    conda create -n e2r python=3.12.3
    conda activate e2r
    pip install numpy==1.26.4 pandas matplotlib tdqm owlready
    # for install pytorch (version 2.2.1), each device has its own way. Seek help here https://pytorch.org/
    # Then install pykeen later
    pip install pykeen
    ```

3. Parse and Spliting data.

    This step will parse `.owl` file into 3 `.tsv` files `train.tsv`, `validate.tsv`, and `test.tsv`

    For RiceDO

    ```bash
    cd ricedo/data
    python parse_and_split_data.py
    cd ../..
    ```

    For TreatO

    ```bash
    cd treato/data
    python parse_and_split_data.py
    cd ../..
    ```

4. After this if you want to use any dataset, move your terminal to the dataset you want to work with (either `ricedo` or `treato`).

    ```bash
    cd ricedo
    # cd treato
    ```

### QE

#### Train QE

*Make sure you move your terminal inside the dataset folder and finished all previous steps.*

You can setup the hyperparameters of QE in the `reasonE.train.py` file. Then, you can train for QE using this command.

```bash
python reasonE.train.py
```

After training, make sure you adjust the hyperparameters in the `reasonE.test.py`. You can evaluate created embedding using this command.

```bash
python reasonE.test.py
```

#### Look though the generated embedding

This command will generate the plot of embedding that I investigate on.

```bash
python plot_embedding.py
```

### Other methods (TransE, ComplEx, TransH, DistMult, ProjE)

The command below will generate embedding in numerious methods using pykeen.

```bash
python trainpykeen.py
```
