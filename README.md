# Cog-Lightweight-OpenPose
Cog wrapper for [Lightweight OpenPose](https://arxiv.org/abs/1811.12004). This is an implementation of the original work's [GitHub repository](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch), see Replicate [model page](https://replicate.com/alaradirik/lightweight-openpose) for the API and demo.


## Basic Usage

To run a prediction:

```bash
cog predict -i image=@sample.png -i image_size=256
```

To start your own server:

```bash
cog run -p 5000 python -m cog.server.http
```

## References
```
@inproceedings{osokin2018lightweight_openpose,
    author={Osokin, Daniil},
    title={Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose},
    booktitle = {arXiv preprint arXiv:1811.12004},
    year = {2018}
}
```
