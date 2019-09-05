# A Neural Sign Language Translation Model with Improved Encoder

This repo contains the training and evalution code of Sign2Text setup for translation sign langauge videos to spoken language sentences. 

This code is based on [a baseline of Cihan Camgoz et al.'s Neural Sign Translation](https://github.com/neccam/nslt). 

## Requirements
* Python 3
* Tensorflow Version >= 1.3.0

## Usage

### Preparation
* (Dataset) Download and extract [RWTH-PHOENIX-Weather 2014 T: Parallel Corpus of Sign Language Video, Gloss and Translation](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) 
* Then you can run [`python3 ResizeImages.py`](ResizeTool/ResizeImages.py) from the folder [ResizeTool](ResizeTool)  to resize the images to 227x227
* Download [AlexNet TensorFlow weights](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) and put it under the folder [BaseModel](BaseModel). You can run [`sh ./download_alexnet_weights.sh`](BaseModel/download_alexnet_weights.sh) from the folder [BaseModel](BaseModel)

### Training Sample Usage (From [nslt_end2end](nslt_end2end) folder)
```
python3 -m nmt --src=sign --tgt=de --train_prefix=Data/phoenix2014T.train --dev_prefix=Data/phoenix2014T.dev --test_prefix=Data/phoenix2014T.test --vocab_prefix=Data/phoenix2014T.vocab --source_reverse=True --residual=True --unit_type=gru --metrics=bleu,rouge --num_units=1000 --num_layers=4 --attention=bahdanau --num_train_steps=200000 --base_gpu=<gpu_id> --num_gpus=1 --eval_on_fly=True --cuda_visible_devices=<gpu_id> --out_dir=<your_output_dir>
```

* You can run [`sh ./train.sh`](nslt_end2end/train.sh) from the folder [nslt_end2end](nslt_end2end) 

### Inference Sample Usage
```
python3 -m nmt --out_dir=<your_model_dir> --inference_input_file=<input_video_paths.sign> --inference_output_file=<predictions.de> --inference_ref_file=<ground_truth.de> --base_gpu=<gpu_id>
```
* You can run [`sh ./infer.sh`](nslt_end2end/infer.sh) from the folder [nslt_end2end](nslt_end2end)


## Reference

Please cite the paper below if you use this code in your research:

    @inproceedings{jiangbin2019nslt,
      author = {Jiangbin Zheng and Yidong Chen and Xiaodong Shi and Suhail Muhammad Kamal},
      title = {A Neural Sign Language Translation Model with Improved Encoder},
      booktitle = {},
      year = {2019}
    }
	
## Acknowledgement
- Thank the author Cihan Camgoz et al.'s code: [nslt](https://github.com/neccam/nslt).
