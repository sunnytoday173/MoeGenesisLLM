from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained("./image/checkpoints/", torch_dtype=torch.float16,
                                                 variant="fp16").to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()


def generate_image(prompt, image_path="bot_avatar.png"):
    image = pipe(prompt=prompt, height=512, width=512, num_inference_steps=25).images[0]
    image.save(image_path)
    return


if __name__ == "__main__":
    prompt = "A cute girl with horse ears wearing japanese clothes"
    generate_image(prompt)

