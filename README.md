```

```

RIDI https://github.com/higerra/ridi_imu https://www.dropbox.com/s/9zzaj3h3u4bta23/ridi_data_publish_v2.zip?dl=0
MotionTransformer http://deepio.cs.ox.ac.uk/
```
wget http://tjunet.top/owncloud/index.php/s/fRzd8yMTjnnTMDa/download -O OIOD.zip
wget http://tjunet.top/owncloud/index.php/s/Y5nNCPFTCiZs5jj/download -O ridi.zip
wget http://tjunet.top/owncloud/index.php/s/gZfD5B29sLekXoi/download -O TJUIMU.zip
unzip OIOD.zip
mv Oxford\ Inertial\ Odometry\ Dataset/ OxIOD
unzip ridi.zip
mv data_publish_v2 ridi
unzip TJUIMU.zip -d tjuimu

export data_path="/home/wjk/Workspace/Datasets"
```


```bash
$ pip install -r requirements.txt
```