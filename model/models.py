import os
import pyaudio
import wave
import torch
import torch.nn as nn
import torchaudio as ta
import numpy as np
from .fbanksampe import fesubsampling
from .PosFeedForward import PositionwiseFeedForward
from .Attention import MultiHeadedAttention


class Layers(nn.Module):
    def __init__(self, attention_heads, d_model, linear_units, residual_dropout_rate):
        super(Layers, self).__init__()

        self.self_attn = MultiHeadedAttention(attention_heads, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, linear_units)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.dropout2 = nn.Dropout(residual_dropout_rate)

    def forward(self, x, mask):
        residual = x
        x = residual + self.dropout1(self.self_attn(x, x, x, mask))
        x = self.norm1(x)

        residual = x
        x = residual + self.dropout2(self.feed_forward(x))
        x = self.norm2(x)

        return x, mask


class speech_model(nn.Module):

    def __init__(self, input_size=40, d_model=320, attention_heads=8, linear_units=1280, num_blocks=12,
                 repeat_times=1, pos_dropout_rate=0.0, slf_attn_dropout_rate=0.0, ffn_dropout_rate=0.0,
                 residual_dropout_rate=0.1):
        super(speech_model, self).__init__()

        self.embed = fesubsampling(input_size, d_model)

        self.blocks = nn.ModuleList([
            Layers(attention_heads,
                   d_model,
                   linear_units,
                   residual_dropout_rate) for _ in range(num_blocks)
        ])
        self.liner = nn.Linear(d_model, 4709)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, inputs):
        enc_mask = torch.sum(inputs, dim=-1).ne(0).unsqueeze(-2)
        enc_output, enc_mask = self.embed(inputs, enc_mask)

        enc_output.masked_fill_(~enc_mask.transpose(1, 2), 0.0)

        for _, block in enumerate(self.blocks):
            enc_output, _ = block(enc_output, enc_mask)
        lin_ = self.liner(enc_output.transpose(0, 1))
        logits_ctc_ = self.softmax(lin_)
        return logits_ctc_


class record_speech():
    def __init__(self, npy="dic.dic.npy", pt="sp_model.pt", filename="temp.wav", filepath="", time=5,
                 device='cpu'):
        self.npy = npy
        self.pt = pt
        self.filename = filename
        self.filepath = filepath
        self.time = time
        self.device = device
        self.model_lo = speech_model()

    def record(self):
        # 检查音频文件名称，若有则删，若无则录
        dictionary_list = os.listdir(os.getcwd())
        if self.filename in dictionary_list: os.remove(self.filename)
        CHUNK = 256
        FORMAT = pyaudio.paInt16
        CHANNELS = 2  # 声道数
        RATE = 16000  # 采样率
        RECORD_SECONDS = self.time
        WAVE_OUTPUT_FILENAME = self.filepath + self.filename
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("*" * 10, "开始录音：请在" + str(self.time) + "秒内输入语音")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("*" * 10, "录音结束\n")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    def get_fu(self):
        _wavform, _ = ta.load_wav(self.filename)
        _feature = ta.compliance.kaldi.fbank(_wavform, num_mel_bins=40)
        _mean = torch.mean(_feature)
        _std = torch.std(_feature)
        _T_feature = (_feature - _mean) / _std
        inst_T = _T_feature.unsqueeze(0)
        return inst_T

    def main(self):
        inst_T = self.get_fu()
        log_ = self.model_lo(inst_T)
        _pre_ = log_.transpose(0, 1).detach().numpy()[0]
        liuiu = [dd for dd in _pre_.argmax(-1) if dd != 0]
        npy_path = 'models/' + self.npy
        # num_wor = np.load('models/dic.dic.npy').item()
        num_wor = np.load(npy_path).item()
        str_end = ''.join([num_wor[dd] for dd in liuiu])
        return str_end

    def recognition(self):
        device_ = torch.device(self.device)
        pt_path = 'models/' + self.pt

        self.model_lo.load_state_dict(torch.load(pt_path, map_location=device_))
        self.model_lo.eval()
        result_ = self.main()
        print('识别结果是： ', result_)
