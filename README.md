# All You Need to Know About Image Retrieval

This is a repo to easily run experiment with 4 retrieval datasets, over 30 losses, multiple miners, etc.
All the magic happens thanks to the amazing [PyTorch Metric Learning Library](https://kevinmusgrave.github.io/pytorch-metric-learning/), which has all these things implemented.

Just run
```
python main.py --dataset CUB --loss NTXentLoss --batch_size 64 --sampler_m 2
```
and it will automatically download the CUB dataset, and run the training with an NT-Xent loss, a batch size of 64 with M=2 (i.e. 2 images per class).

Running `python main.py -h` will show you all the available parameters to choose from.

Despite being quite powerful (allows to run thousands of different experiments), this repo is super simple, with less than 400 lines of code in total across all python files.
