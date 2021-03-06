# TST-Lite for Anomaly Detection


TST-Lite ð is a better and more efficient way to detect anomalies in time series data with Attention-based Learning.


## Architecture

![Architecture of TST-Lite.png](asset/img.png)

## Directory Structure

```shell
.
âââ Dockerfile              # DockerfileæåDockeréå
âââ README.md               # è¯´æææ¡£
âââ app.py                  # Flaskå¥å£
âââ asset
âÂ Â  âââ img.png             # TST-Lite æµç¨å¾
âââ dataset                 # æ°æ®é
âââ model                   # æ¨¡åç®å½
âââ requirements.txt        # ä¾èµå
âââ self_check.py           # å®ç°èªæ£åè½
âââ train.py                # è®­ç»æ¨¡åçå°è£
âââ utils                   # å·¥å·
    âââ data_prepare.py     # æ°æ®åå¤
    âââ eval.py             # è¯ä¼°æ¨¡å
    âââ for_overview.py     # æåæ¦è¿°ä¿¡æ¯
    âââ plot_and_loss.py    # ç»å¶ loss å¾å
    âââ train.py            # è®­ç»æ¨¡å
```



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

