from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained("./image/checkpoints/", torch_dtype=torch.float16,
                                                 variant="fp16").to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()


def generate_image(prompt, negative_prompt, image_path="bot_avatar.png"):
    image = pipe(prompt=prompt,negative_prompt=negative_prompt,height=512, width=512, num_inference_steps=25).images[0]
    image.save(image_path)
    return


if __name__ == "__main__":
    prompt = "A cute girl with horse ears wearing japanese clothes"
    example_prompt = "(masterpiece),(highest quality),highres,(an extremely delicate and beautiful),(extremely detailed), 1girl, japanese clothes, solo, kimono, animal ears, horse ears, hair over one eye, brown hair, flower, purple eyes, sash, smile, obi, blush, wide sleeves, white kimono, rose, blue flower, blue rose, floral print, closed mouth, horse girl, long sleeves, print kimono, cherry blossoms, looking at viewer, upper body, petals, hair flower, hat, bangs, blurry, arm up, blurry foreground, tilted headwear, depth of field, hair ornament, short hair, one eye covered, hand up, eyebrows visible through hair, long hair, blurry background, pink flower, alternate costume"
    example_negative_prompt = "watercolor, oil painting, photo, deformed, realism, disfigured, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    generate_image(prompt,example_negative_prompt)

