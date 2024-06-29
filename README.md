FedL (Marvel)
======

The research leading to these results has received funding from the European Union’s Horizon 2020
research and innovation programme under grant agreement No 957337.

# Requirements

Use either pip or conda:

```
pip install -r requirements.txt

-or-

conda env create -f marvel-conda-env.yaml
```

# Running the code

```
# Run the server:

python server.py # By default it uses our novel Non-uniform sampling strategy

# Run five clients:

bash run-clients.sh

```

# Extending the code (custom model/dataset)

Replace `mnist.py` file used by the clients for training with your custom dataset or DL model. 

# Defaults

- MNIST dataset
- small FCN, PyTorch LTS
- port for server is 8080
