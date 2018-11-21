# Denoiser build with pytorch

## 安装前置
pytorch == 1.0(unstable)

tensorboardX

lmdb

## 运行方法

### 第一步，生成训练数据
#### Train400
下载方式Mega盘：https://mega.nz/#F!La4AiYII!rnbw-RCg4bpJGVd2T7vnTA

在generate_patches.py调用generate_patches_from_file()

python generate_patches.py --src_dir=./data/Train400 --save_dir=./data/Train400/train --patch_size=40 --stride=10 --sigma=25

#### SIDD_Small
在generate_patches.generate_patches_list()

python generate_patches.py --src_dir=./data/SIDD_Small --save_dir=./data/SIDD_Small/train --patch_size=128 --stride=64

### 第二步，运行训练代码
目前效果还行的模型：https://mega.nz/#!2KJiXSTA!HrX5cCLJGbwVogYmHZp4_5cQUlOkCBqYW51XCjsHPZk
#### Train
python main.py --phase=train --train_set=./data/Train400/train/ --batch_size=128 --lr=0.001
#### Evaluate
python main.py --phase=eval --checkpoint=./checkpoint/train_DnCNN_9.ckpt --eval_set=./data/Train400/train/ --batch_size=128 --lr=0.001 --is_gray=true
#### Inference
python main.py --phase=inference --checkpoint=./checkpoint/train_DnCNN_9.ckpt --test_image=test.png --is_gray=true