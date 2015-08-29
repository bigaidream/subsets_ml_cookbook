<!-- toc -->


# Neon with Spearmint for Hyperparameter Optimization

> start: 28 August, 2015

I've been trying to install `neon` on my machines (one with Intel i7, 32G RAM, Nvidia GeForce GT 630 and Nvidia Tesla K40c, Ubuntu `14.04.2`, another one with Intel i7, 64G RAM, Nvidia Titan Black, Ubuntu `14.04.3`) for the last few days, but still failed. 

> It's very weird that Ubuntu `14.04.3` does NOT work with Nvidia GT 630 and CUDA 7.0, in that after installing CUDA 7.0, Ubuntu cannot boot normally. However, it works well with Ubuntu `14.04.2`, which is why I use slightly different Ubuntu versions on two machines. 

It should be noted that `theano` works just fine on my machines. In order to make hyperparameter tuning practical, I'm forced to use CPU mode and run it on a **HUGE** CPU cluster. Luckily, I was awarded some Microsoft Azure computing resources.  

## Entry point

> Official doc on hyperopt: http://neon.nervanasys.com/docs/latest/hyperparameter_tuning.html

The very initial entry point is `/neon/bin/hyperopt.py`, and it calls `/spearmint/spearmint/bin/spearmint`, which is a `shell` script. 

We can run:
```bash
bash -x spearmint
```
This will show how the `spearmint.sh` actually runs, i.e. every step it takes. It turns out that it calls `/spearmint/spearmint/main.py`. 

## How Spearmint runs

> https://github.com/JasperSnoek/spearmint

```python
python main.py --driver=local --method=GPEIOptChooser --method-args=noiseless=1 ../hyperopt_experiment/spear_config.pb
```

If we take a look at `spear_config.pb`, it reads:
```
name: "neon.hyperopt.gen_yaml_and_run"
```
It seems that in order to modify it to tune other learning algorithms, say `keras`, I need to modify `neon.hyperopt.gen_yaml_and_run`. 

## Debugging Neon and Spearmint within Python
In `Pycharm`, I set `script parameters` of `/spearmin/spearmint/main.py` to be `--driver=local --method=GPEIOptChooser --method-args=noiseless=0 --polling-time=20 --max-concurrent=2 -w --port=50000 /home/jie/Documents/hyperopt_experiment/spear_config.pb`, thus it is easy to debug, e.g. setting break-points. 

> Of course, I can attach `Pycharm` to a running python process, but it's not that convenient. As far as I know, there is no easy way to attach the python process at the very beginning. 

## `gen_yaml_and_run.py` 

`gen_yaml_and_run.py` seems to be the main modification made by `neon` to work with `spearmint`. 