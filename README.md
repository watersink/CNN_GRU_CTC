# CNN_GRU_CTC
基于传统多标签的定长验证码识别和基于GRU+CTC的不定长验证码识别

## original cnn multiple lables classification
    cd ./cnn

### train
    python3 train_cnn.py
### test
    python3 eval_test_cnn.py
![image]( https://github.com/watersink/CNN_GRU_CTC/raw/master/result/cnn.jpg)

## gru+ctc
    cd ./ctc

### train
    python3 train_ctc.py
### test
    python3 eval_test_ctc.py
<div>
<img width="300" height="300" src="https://github.com/watersink/CNN_GRU_CTC/raw/master/result/ctc1.jpg"/>
<img width="300" height="300" src="https://github.com/watersink/CNN_GRU_CTC/raw/master/result/ctc.jpg"/>
</div>

## references

https://github.com/fchollet/keras/blob/master/examples/image_ocr.py

https://github.com/ypwhs/captcha_break

https://github.com/baidu-research/warp-ctc

## ocr detection and recognition group 
![image]( https://github.com/watersink/CNN_GRU_CTC/raw/master/images/OCR.png) 
