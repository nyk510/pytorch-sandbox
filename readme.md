# pytorch starter

pytorch でいろいろ遊んでみるリポジトリ

## Requirements

* pytorch: 0.40

## 環境の準備

### docker による環境構築

cuda 9.1 / cudnn 7 (ubuntu16.04) の docker 環境を構築します。
イメージ内では miniconda 上に pytorch と諸々がインストールされます。

```bash
docker pull nvidia/cuda:9.1-cudnn7-runtime
docker build -t pytorch-cuda9.1 ./docker
nvidia-docker run -it -v $PWD:/workdir
```