# SNU x Morpheus 3D Wrinkle Detetion

최종적으로 두가지 method를 제안하였는데 두가지 메소드가 각각 wbce_model 폴더와 partitioned_model 폴더에 들어가 있습니다.

## wbce_model (Weighted Binary Cross Entropy U-net)

1248x1344로 crop한 이미지들에 대해 Weighted binary cross entropy loss 를 사용해 학습시킨 모델의 관련 코드들이 이 폴더에 들어가 있습니다. 이 모델의 경우 제공받은 24G memory의 GPU에서 학습을 하였는데요. 11G memory의 GPU에서 학습시키려고 한 결과 메모리 부족으로 터지는 것을 확인할 수 있었습니다. 따라서 memory 크기가 충분히 큰 GPU에서 학습 가능한 method입니다.

사용된 data는 data/final_data 에 저장되어 있습니다.

```python main.py``` 로 실행시키면 기본 설정값으로 학습이 진행됩니다.

설정할 수 있는 파라미터들을 보고 싶으시면 ```python main.py --help```를 사용하시면 설정 가능한 옵션들을 볼 수 있습니다.
대표적으로 loss function을 설정하기 위해 ```python main.py --loss f1```, ```python main.py --loss weighted_BCE``` 와 같이 실행시킬 수 있습니다.

위와 같은 코드로 학습을 시키면 log 폴더에 자동으로 학습시간_loss(ex 12-16_00-32_weighted_BCE)의 형태로 폴더가 생성되고 log가 기록됩니다. tsv파일 형식으로 acc/precision/recall/f1 등을 기록하였습니다. 또한 tensorboard를 이용해서도 logging 하였습니다.
tensorboard를 이용해 log를 보고 싶으시다면 ```tensorboard --logdir ./log/11-01_08-59_weighted_BCE/```와 같이 실행시키시면 됩니다.

또한 학습시에 checkpoint 폴더에 학습 중간에 model checkpoint를 기록을 하게됩니다.

따라서 이 checkpoint를 가져와서 validation images들에 대해 evaluate를 할 수 있도록 구현하였습니다.

```python evaluate.py --ckpt_dir ./checkpoint/11-01_08-59_weighted_BCE --ckpt_name model_epoch100.pth```
와 같이 실행을 하면 해당 checkpoint 폴더의 checkpoint를 불러와 validation dataset에 대해 돌려보고 이때 f1 score 기록 및 실제 모델이 예측한 값을 npy 및 png 파일 형태로 result 폴더에 저장하게 됩니다.
만약 ```--ckpt_name``` 옵션을 생략한다면 가장 epoch이 높을 때 저장된 checkpoint를 불러와 evaluate 합니다.

이 메소드에 대해 가장 높은 f1 score를 보인 checkpoint를  ./wbce_model/checkpoint/12-16_18-13_weighted_BCE 폴더에 두었습니다.

## partitioned model
* `partitioned_model.ipynb`
  * 얼굴을 부위 별로 나누어 학습하고 합치는 모든 과정 포함
  * Interactive python notebook 파일
  * 셀 하나 씩 실행하면서 각각의 설명이 내부적으로 있기 때문에 자세한 구현 설명은 생략
* `checkpoint`
  * model의 checkpoint를 저장 (마지막 epoch이 종료되고 저장되는 옵션)
  * 현재 GeForce GTX 1080Ti 4 cores로 학습시킨 모델의 checkpoint가 저장되어 있음 (200 epochs, 16 batches)
* `log`
  * model의 log history를 저장
  * 현재 GeForce GTX 1080Ti 4 cores로 학습시킨 모델의 log가 저장되어 있음 (200 epochs, 16 batches)
* `result`
  * 모델을 이용하여 시각화한 결과
* `unet`
  * unet baseline modules
* `plot_metric.py`
  * metric plot modules

## preprocessing
데이터 전처리에 사용된 코드들입니다.
