# TST-Lite for Anomaly Detection


TST-Lite 🚀 is a better and more efficient way to detect anomalies in time series data with Attention-based Learning.


## Architecture

![Architecture of TST-Lite.png](asset/img.png)

## Quick Start

1. first, run the following command to generate `requirements.txt`
```shell
pipreqs ./ --encoding=utf8 --force
```


2. install essential packages
```shell
pip install -r requirements.txt
```

3. change the config for your environment

4. run the `app.py`
```shell
python app.py
```

## Docker
```shell
docker run -it -d -p 5001:5001 --name tst-lite cocoshe/ts:api
```

## Link backend services api
Use golang for some services routes, click [here](https://github.com/cocoshe/yuheng)

