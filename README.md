# Flow-to-depth (FDNet) video-depth-estimation
This is the implementation of paper

Video Depth Estimation by Fusing Flow-to-Depth Proposals

Jiaxin Xie, [Chenyang Lei](https://chenyanglei.github.io/), Zhuwen Li, [Li Erran Li](http://www.cs.columbia.edu/~lierranli/), [Qifeng Chen](https://cqf.io/)

In IROS 2020.

See our paper (https://arxiv.org/pdf/1912.12874.pdf) for more details. Please contact Jiaxin Xie (jxieax@ust.cse.hk) if you have any questions.


## Prerequisites
This codebase was developed and tested with Tensorflow 1.4.0 and Numpy 1.16.2

## Evaluation on KITTI Eigen Split
IF you want to generate GroundTruth Depth from KITTI RAW data,  download KITTI dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website.

Meanwhile, we also provided GroundTruth Depth save in npy file,  download it from [here](https://drive.google.com/file/d/18WNghzKudLUXIbkrcf8YqH1N8Lpjqr4R/view?usp=sharing) 

Our final results on KITTI Eigen is availible on [here](https://drive.google.com/file/d/18WNghzKudLUXIbkrcf8YqH1N8Lpjqr4R/view?usp=sharing)

Then run

```bash
python kitti_eval/eval_depth_general.py --kitti_dir=/path/to/raw/kitti/dataset/ or /path/to/downloaded/GoundTruth/npy/file/ --pred_file=/path/to/our/final/results/
```
