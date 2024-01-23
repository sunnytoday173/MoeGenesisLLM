import numpy as np
import os
import soundfile as sf
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# 将子文件夹的路径添加到sys.path中
# 看下来是目前实现成本最低的方案
sys.path.append(os.path.abspath('audio/fishspeech'))
print(os.path.abspath('audio/fish-speech'))


from audio.fishspeech.tools.llama.generate import encode_tokens,generate,decode_one_token

# 载入音频模型
from audio.fishspeech.tools.llama.generate import load_model
device = "cuda"
half = True
precision = torch.half if half else torch.bfloat16
print("Loading model ...")

t2s_config_name = "text2semantic_finetune"
t2s_checkpoint_path = "./audio/checkpoints/text2semantic-400m-v0.3-4k.pth"
t2s_model = load_model(t2s_config_name, t2s_checkpoint_path, device, precision)
torch.cuda.synchronize()

from audio.fishspeech.tools.vqgan.inference import load_model
vqgan_config_name = "vqgan_pretrain"
vqgan_checkpoint_path = "./audio/checkpoints/vqgan-v1.pth"
vqgan_model = load_model(vqgan_config_name, vqgan_checkpoint_path)

def encode_text(tokenizer,text,prompt_text,prompt_tokens,device="cuda",use_g2p=True,speaker=None,order="zh,jp,en",compile=True,num_samples=1,max_new_tokens=0,temperature=0.7,top_k=None,top_p=0.5,repetition_penalty=1.5,turn=0,output_path=None):
    encoded = encode_tokens(
        tokenizer,
        text,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
        bos=True,
        device=device,
        use_g2p=use_g2p,
        speaker=speaker,
        order=order,
    )
    prompt_length = encoded.size(1)
    print(f"Encoded prompt shape: {encoded.shape}")
    if compile:
        global decode_one_token
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )
    
    for i in range(num_samples):
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        y = generate(
            model=t2s_model,
            prompt=encoded,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            precision=precision,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        if i == 0 and compile:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

        torch.cuda.synchronize()
        t = time.perf_counter() - t0

        tokens_generated = y.size(1) - prompt_length
        tokens_sec = tokens_generated / t

        codes = y[1:, prompt_length:-1]
        codes = codes - 2
        assert (codes >= 0).all(), "Codes should be >= 0"
        
        if output_path is None:
            output_path = f"./audio/record/codes_{turn}_{i}.npy"

        np.save(output_path, codes.cpu().numpy())
        print(f"Saved codes to {output_path}")
    return None

def generate_audio(input_path,output_path):
    indices = np.load(input_path)
    indices = torch.from_numpy(indices).to(vqgan_model.device).long()
    
    # Restore
    indices = indices.unsqueeze(1).unsqueeze(-1)
    mel_lengths = indices.shape[2] * (
        vqgan_model.downsample.total_strides if vqgan_model.downsample is not None else 1
    )
    mel_lengths = torch.tensor([mel_lengths], device=vqgan_model.device, dtype=torch.long)
    mel_masks = torch.ones(
        (1, 1, mel_lengths), device=vqgan_model.device, dtype=torch.float32
    )

    text_features = vqgan_model.vq_encoder.decode(indices)

    print(
        f"VQ Encoded, indices: {indices.shape} equivalent to "
        + f"{1/(mel_lengths[0] * vqgan_model.hop_length / vqgan_model.sampling_rate / indices.shape[2]):.2f} Hz"
    )

    text_features = F.interpolate(text_features, size=mel_lengths[0], mode="nearest")

    # Sample mels
    decoded_mels = vqgan_model.decoder(text_features, mel_masks)
    fake_audios = vqgan_model.generator(decoded_mels)
    print(
        f"Generated audio of shape {fake_audios.shape}, equivalent to {fake_audios.shape[-1] / vqgan_model.sampling_rate:.2f} seconds"
    )

    # Save audio
    fake_audio = fake_audios[0, 0].cpu().detach().numpy().astype(np.float32)
    
    sf.write(output_path, fake_audio, vqgan_model.sampling_rate)
    print(f"Saved audio to {output_path}")

if __name__ == "__main__":
    text = "今天天气真不错，我们去野餐吧"
    t2s_tokenizer_path = "./tokenizer"
    audio_tokenizer = AutoTokenizer.from_pretrained(t2s_tokenizer_path)
    prompt_text = "你好，我是派蒙"
    prompt_tokens_path = "./paimon.npy"
    prompt_tokens = (
        torch.from_numpy(np.load(prompt_tokens_path)).cuda()
        if prompt_tokens_path is not None
        else None
    )
    encode_text(audio_tokenizer,text, prompt_text, prompt_tokens,turn=0)

