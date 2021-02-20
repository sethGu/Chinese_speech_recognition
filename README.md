# Chinese_speech_recognition 中文语音识别
windows可用，准确率相对较高。

## 环境配置
推荐使用python venv或conda，本人使用conda
- Python 3.7.7
- conda 4.8.3
- pip 21.0.1

## 使用方法
1. git clone至本地
2. cd Chinese_speech_recognition
3. pip install -r requirements.txt
4. python demo.py
- demo.py是实时录音，
- recording_recognition.py是识别已有录音，在此项目中是temp.wav，
- 对于已有录音的识别，要求音频为wav文件，且采样率为16000（亲测声道数为2不会出问题）。

## 参数调整
若想调整一些录音时的参数，可以对在model/models/record_speech()中做如下修改：
- 更改录音时长，参数为record_speech(*time = 5*)   int型
- 更改保存的wav音频文件名称，参数为record_speech(*filename="temp.wav"*)    str型
- 更改保存的wav音频文件路径，参数为record_speech(*filepath=""*)    str型
- 更改设备配置，参数为record_speech(*device='cpu'*)    str型，目前只有用cpu的
- 更改语义模型，参数为record_speech(*pt="sp_model.pt"*)   str型

## 与github其他项目相比的优势
尝试过一些github上其他的语音识别项目，如MASR、ASRT等，但由于ctcdecode在windows上的使用非常之麻烦，且使用虚拟机如WSL之类又较为费时，因此体验较差。除此之外，不是每台设备都有强大的GPU，但每台设备都有CPU，因此此项目的适用性非常广泛，但凡你的CPU不是古董，都可以使用此项目进行较为舒爽的语音识别体验，对一些语音识别不是特别严苛的工业项目也可以使用，且项目存在优化空间。<br>
### 优化
- 可以对model/models.py进行逻辑上的调整；
- 可以搞一些精准度更高的语义模型。
