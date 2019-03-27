# style-transfer
Experimenting with Style Transfer and PyTorch. Inspired by https://towardsdatascience.com/neural-style-transfer-series-part-2-91baad306b24

# Running
I run in docker:
```
sudo docker run --rm -it --runtime=nvidia -v ~/.torch/:/root/.torch/ -w $PWD -v $PWD:$PWD -it pytorch/pytorch:latest
```
