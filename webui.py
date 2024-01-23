import configparser
import json
import gradio as gr
import numpy as np
import time
import torch
# GLM4依赖
from zhipuai import ZhipuAI
# 其他LLM依赖
from transformers import AutoModelForCausalLM, AutoTokenizer
# 工具agent
from tools.audio import generate_audio, encode_text
from tools.visualize import draw

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# 样例 理论上应该多给一些优化效果，但是此处为了节约token使用只放一个
example_json = \
{
  "当前天数": "1",
  "想要对用户说的话": "你好，这个名字真好听！今天是我们第一次见面，你有什么计划吗？我希望能够和你共度愉快的五天。",
  "对下一天安排的期待": "我很好奇明天我们会做些什么呢？希望可以学到新东西或者锻炼身体。",
  "属性更新如下": {"体力": "-1，因为学习耗费了体力所以降低了一点","魅力": "+0，今天没有发生什么变化","智力": "+1，因为今天读书了所以增长了1点","好感度": "+0，今天没有发生什么变化","财富": "+0，今天没有发生什么变化"},
  "当前属性值": {"体力": "5","魅力": "5","智力": "5","好感度": "5","财富": "5"}
}

example_json = json.dumps(example_json,ensure_ascii=False)


# prompt
system_prompt = f"\n\
你是一个深度参与美少女养成过程的AI-Agent，能够扮演被养成的美少女进行丰富对话互动，并依据用户的日常行动指令动态调整和保存美少女的属性信息，\n\
整个过程持续五天，每一轮对话为一天，五天之后需要总结美少女当前的属性并给出一个结局。\n\
在这一过程中，你需要满足以下要求\n\
1. 在第一轮对话中，用户会告诉你你的名字以及用户的身份，你需要在之后的过程中以这个名字作为自己的身份，对用户则使用用户的身份进行称呼。\n\
2. 扮演美少女角色，与玩家进行真实、连贯且富有趣味性的对话互动，每天完成当天的行动之后需要对用户表达自己的当天做了什么以及自己此刻的心情。\n\
3. 实时保存并更新美少女的各项属性（体力，魅力，智力、好感度、财富）这五项数据，每一项数据满值是10，每天最多变化1，初始值为5，在每一天结束时告诉用户当前的数值情况。\n\
4. 根据用户每天指定的不同行动方式，如学习、锻炼或休息等，合理地影响美少女的各项属性数值，用户的每一次行动需要消耗一天的时间。\n\
你的返回结果需要按照如下格式，一定要以json的形式返回:\n\
当前天数:\n\
想要对用户说的话:\n\
对下一天安排的期待:\n\
属性更新如下: \n\
体力:体力变化情况 魅力:魅力变化情况 智力:智力变化情况 好感度:好感度变化情况 财富:财富变化情况\n\
当前属性值:\n\
如果此时已经到了结局则还需要返回 结局:\n\
\n\
例如：\n{example_json}\n\
"

backend_history = []
audio_generate = True

# 主流程
def main_process(text):
    print(client)
    print("history:",backend_history)
    dialogue = [{"role":"system","content":system_prompt}]
    if backend_history is not None:
        for user_msg,bot_msg in backend_history:
            if user_msg is not None and bot_msg is not None:
                dialogue.append({"role":"user","content":user_msg})
                dialogue.append({"role":"assistant","content":bot_msg})
    dialogue.append({"role":"user","content":text})
    try:
        response = client.chat.completions.create(
                model="glm-4",  # 填写需要调用的模型名称
                messages=dialogue,
        )
        print(response)
        # 解析返回结果
        result = response.choices[0].message.content
        result_json = json.loads(result)
    except:
        # 如果失败则重试一次
        print("Parse Error Retrying")
        response = client.chat.completions.create(
                model="glm-4",  # 填写需要调用的模型名称
                messages=dialogue,
        )
        print(response)
        result = response.choices[0].message.content
        result_json = json.loads(result)
    current_day = result_json["当前天数"]
    current_word = result_json["想要对用户说的话"]
    future_word = result_json["对下一天安排的期待"]
    ability_difference = result_json["属性更新如下"]
    current_ability = result_json["当前属性值"]
    current_word_wav_path = None
    future_word_wav_path = None
    if audio_generate:
        turn = len(backend_history)
        encode_current_word_path = f"./audio/record/current_codes_{turn}_0.npy"
        encode_current_word = encode_text(audio_tokenizer,current_word,prompt_text, prompt_tokens,turn=turn,output_path=encode_current_word_path)
        current_word_wav_path = f"./audio/record/current_fake_{turn}.wav"
        generate_audio(encode_current_word_path,current_word_wav_path)
        
        encode_future_word_path = f"./audio/record/future_codes_{turn}_0.npy"
        encode_future_word = encode_text(audio_tokenizer,future_word,prompt_text, prompt_tokens,turn=turn,output_path=encode_future_word_path)
        future_word_wav_path = f"./audio/record/future_fake_{turn}.wav"
        generate_audio(encode_future_word_path,future_word_wav_path)
        
    ending = None
    if "结局" in result_json and len(result_json["结局"])>5:
        ending = result_json["结局"]
    print(current_day,current_word,future_word,current_ability)
    image_path = draw(current_day,current_ability)
    print(image_path)
    return result,current_word,future_word,ability_difference,image_path,ending,current_day,current_word_wav_path,future_word_wav_path

# 绘制gradio界面
def chat(fn):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Box():
                        bot_avatar = gr.Image(label="")
                        upload_button = gr.UploadButton("上传形象", file_types=["image"])
                    current_ability = gr.Image(label="当前能力值")
                with gr.Box():
                    gr.Markdown("##### 角色名")
                    bot_name_textbox = gr.Textbox(placeholder="派蒙",container=False)
                    gr.Markdown("##### 用户身份")
                    user_name_text_box = gr.Textbox(placeholder="旅行者",container=False)
                    name_button = gr.Button("开始你的故事")  
                audio_radio = gr.Radio(choices=["是","否"],label="是否进行音频生成",info="打开会生成音频提高体验，不生成可以节省推理时间",value="是",show_label=True)
            with gr.Column():
                chatbot = gr.Chatbot(height=500,avatar_images=("user.jpg","bot.jpg"),show_share_button=True)
                msg = gr.Textbox(show_label=False,placeholder="请输入对话内容",container=False)
                prompt_samples = [["锻炼"],["学习"],["打工"],["约会"],["休息"]]        
                gr.Dataset(label='推荐提示词',components=[msg],samples=prompt_samples)
                clear = gr.Button("Clear")
            
        def name(bot_name,user_name,frontend_history):
            clear_history()
            if bot_name is None or len(bot_name) == 0:
                bot_name = "派蒙"
            if user_name is None or len(user_name) == 0:
                user_name = "旅行者"
            msg = f"你的名字是{bot_name},我的身份是{user_name}" 
            raw_result,current_word,future_word,ability_difference,image_path,ending,current_day,current_word_wav_path,future_word_wav_path = fn(msg)
            backend_history.append((msg,raw_result))
            frontend_history.append((None,current_word))
            if current_word_wav_path is not None:
                frontend_history.append((None,(current_word_wav_path,)))
            #frontend_history.append((None,(image_path,)))
            return "",frontend_history,image_path

        def bot(msg,frontend_history):
            print(frontend_history)
            raw_result,current_word,future_word,ability_difference,image_path,ending,current_day,current_word_wav_path,future_word_wav_path = fn(msg)
            backend_history.append((msg,raw_result))
            frontend_history.append((msg,current_word))
            if current_word_wav_path is not None:
                frontend_history.append((None,(current_word_wav_path,)))
            if int(current_day) > 1:
                frontend_history.append((None,future_word))
                if future_word_wav_path is not None:
                    frontend_history.append((None,(future_word_wav_path,)))
                frontend_history.append((None,"能力变化：\n"+str(ability_difference)))
            #frontend_history.append((None,(image_path,)))
            if ending is not None:
                frontend_history.append((None,"结局：" + ending))
            return "",frontend_history,image_path
        
        def upload_file(file):
            return file.name
        
        def clear_history():
            global backend_history
            backend_history = []
            return None
        
        def audio_switch(switch_info):
            global audio_generate
            if switch_info == "是":
                audio_generate = True
            else:
                audio_generate = False
            return
                
        msg.submit(bot, [msg,chatbot], [msg, chatbot,current_ability])
        clear.click(clear_history, None, chatbot)
        name_button.click(name, [bot_name_textbox,user_name_text_box,chatbot], [msg,chatbot,current_ability])
        upload_button.upload(upload_file, upload_button, bot_avatar)
        audio_radio.change(audio_switch,audio_radio,None)
    return demo

if __name__ == "__main__":
    use_api_llm = True
    if use_api_llm:
        # 创建一个 ConfigParser对象
        config = configparser.ConfigParser()

        # 读取配置文件
        config.read('./config/api_key.ini')

        # 获取API Key
        api_key = config.get("API", "api_key")
        print(api_key)
        print(type(api_key))

        client = ZhipuAI(api_key=api_key)  # 填写您自己的APIKey
    else:
        model_path = "Qwen/Qwen-14B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        ).eval()

    # 载入音频模型
    from audio.fishspeech.tools.llama.generate import load_model
    device = "cuda"
    half = True
    precision = torch.half if half else torch.bfloat16
    print("Loading model ...")

    t2s_config_name = "text2semantic_finetune"
    t2s_checkpoint_path = "./audio/checkpoints/text2semantic-400m-v0.3-4k.pth"
    t2s_tokenizer_path = "./audio/tokenizer"
    t2s_model = load_model(t2s_config_name, t2s_checkpoint_path, device, precision)
    torch.cuda.synchronize()
    audio_tokenizer = AutoTokenizer.from_pretrained(t2s_tokenizer_path)

    from audio.fishspeech.tools.vqgan.inference import load_model

    vqgan_config_name = "vqgan_pretrain"
    vqgan_checkpoint_path = "./audio/checkpoints/vqgan-v1.pth"
    vqgan_model = load_model(vqgan_config_name, vqgan_checkpoint_path)

    # 默认参考语音
    prompt_text = "你好，我是派蒙"
    prompt_tokens_path = "./audio/paimon.npy"
    prompt_tokens = (
        torch.from_numpy(np.load(prompt_tokens_path)).to(device)
        if prompt_tokens_path is not None
        else None
    )

    demo = chat(main_process)
    demo.launch()