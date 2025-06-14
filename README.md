<p align="center">
<h1 align="center"><strong>TC-Light: Temporally Consistent Relighting for Dynamic Long Videos</strong></h1>
  <p align="center">
    <em>Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences</em>
  </p>
</p>

<div id="top" align="center">

[![](https://img.shields.io/badge/%F0%9F%9A%80%20Project-V1-green)](https://dekuliutesla.github.io/citygs/)
[![](https://img.shields.io/badge/%F0%9F%9A%80%20Project-V2-blue)](https://dekuliutesla.github.io/CityGaussianV2/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/TeslaYang123/CityGaussian)
![GitHub Repo stars](https://img.shields.io/github/stars/DekuLiuTesla/CityGaussian)

</div>
<p align="center">
  <img src="imgs/teaser.png">
</p>

This repo contains official implementations of **TC-Light**, a one-shot model used to manipulate the illumination of **high-dynamic videos** such as motion-rich actions and frequent switch of foreground and background objects. It is distinguished by:

- Outstanding Temporal Consistency on Highly Dynamic Scenarios.
- Superior Computational Efficiency that Enables Long Video Processing (can process 300 frames with resolution of 1280x720 on 40G A100).

These features make it particularly suitable for sim2real and real2real augmentation for Embodied Agents or preparing video pairs to train stronger video relighting models. Star ⭐ us if you like it!


# 🛠 Getting Started

Install the required environment as follows:
```bash
git clone https://github.com/lllyasviel/IC-Light.git
cd TC-Light
conda create -n tclight python=3.10
conda activate tclight
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
As a quick start, you can use:
```bash
# support .mp4, .avi, and folder containing sequential images
# --multi_axis enables decayed multi-axis denoising, which enhances consistency but slow down the diffusion process
python run.py -i /path/to/your/video -p "your_prompt" \
              -n "your_negative_prompt" \  #  optional
              --multi_axis  # optional
```
By default, it will relight the first 30 frames with resolution 960x720. The default negative prompt is adopted from [Cosmos-Transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1). If it is the first-time running on a specific video, it would generate and save flow un the path to your video. 

For a fine-grained control, you can customize your .yaml config file and run:
```bash
python run.py --config path/to/your_config.yaml
```
You can start from [configs/tclight_custom.yaml](configs/tclight_custom.yaml), which records the most frequently used parameters and detailed explanation.

Model downloading is automatic.


# Screenshot

### Text-Conditioned Model

(Note that the "Lighting Preference" are just initial latents - eg., if the Lighting Preference is "Left" then initial latent is left white right black.)

---

**Prompt: beautiful woman, detailed face, warm atmosphere, at home, bedroom**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/87265483-aa26-4d2e-897d-b58892f5fdd7)

---

**Prompt: beautiful woman, detailed face, sunshine from window**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/148c4a6d-82e7-4e3a-bf44-5c9a24538afc)

---

**beautiful woman, detailed face, neon, Wong Kar-wai, warm**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/f53c9de2-534a-42f4-8272-6d16a021fc01)

---

**Prompt: beautiful woman, detailed face, sunshine, outdoor, warm atmosphere**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/25d6ea24-a736-4a0b-b42d-700fe8b2101e)

---

**Prompt: beautiful woman, detailed face, sunshine, outdoor, warm atmosphere**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/dd30387b-0490-46ee-b688-2191fb752e68)

---

**Prompt: beautiful woman, detailed face, sunshine from window**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/6c9511ca-f97f-401a-85f3-92b4442000e3)

---

**Prompt: beautiful woman, detailed face, shadow from window**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/e73701d5-890e-4b15-91ee-97f16ea3c450)

---

**Prompt: beautiful woman, detailed face, sunset over sea**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/ff26ac3d-1b12-4447-b51f-73f7a5122a05)

---

**Prompt: handsome boy, detailed face, neon light, city**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/d7795e02-46f7-444f-93e7-4d6460840437)

---

**Prompt: beautiful woman, detailed face, light and shadow**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/706f70a8-d1a0-4e0b-b3ac-804e8e231c0f)

(beautiful woman, detailed face, soft studio lighting)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/fe0a72df-69d4-4e11-b661-fb8b84d0274d)

---

**Prompt: Buddha, detailed face, sci-fi RGB glowing, cyberpunk**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/68d60c68-ce23-4902-939e-11629ccaf39a)

---

**Prompt: Buddha, detailed face, natural lighting**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/1841d23d-0a0d-420b-a5ab-302da9c47c17)

---

**Prompt: toy, detailed face, shadow from window**

Lighting Preference: Bottom

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/dcb97439-ea6b-483e-8e68-cf5d320368c7)

---

**Prompt: toy, detailed face, sunset over sea**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/4f78b897-621d-4527-afa7-78d62c576100)

---

**Prompt: dog, magic lit, sci-fi RGB glowing, studio lighting**

Lighting Preference: Bottom

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/1db9cac9-8d3f-4f40-82e2-e3b0cafd8613)

---

**Prompt: mysteriou human, warm atmosphere, warm atmosphere, at home, bedroom**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/5d5aa7e5-8cbd-4e1f-9f27-2ecc3c30563a)

---

### Background-Conditioned Model

The background conditioned model does not require careful prompting. One can just use simple prompts like "handsome man, cinematic lighting".

---

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/0b2a889f-682b-4393-b1ec-2cabaa182010)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/477ca348-bd47-46ff-81e6-0ffc3d05feb2)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/5bc9d8d9-02cd-442e-a75c-193f115f2ad8)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/a35e4c57-e199-40e2-893b-cb1c549612a9)

---

A more structured visualization:

![r1](https://github.com/lllyasviel/IC-Light/assets/19834515/c1daafb5-ac8b-461c-bff2-899e4c671ba3)

# Imposing Consistent Light

In HDR space, illumination has a property that all light transports are independent. 

As a result, the blending of appearances of different light sources is equivalent to the appearance with mixed light sources:

![cons](https://github.com/lllyasviel/IC-Light/assets/19834515/27c67787-998e-469f-862f-047344e100cd)

Using the above [light stage](https://www.pauldebevec.com/Research/LS/) as an example, the two images from the "appearance mixture" and "light source mixture" are consistent (mathematically equivalent in HDR space, ideally).

We imposed such consistency (using MLPs in latent space) when training the relighting models.

As a result, the model is able to produce highly consistent relight - **so** consistent that different relightings can even be merged as normal maps! Despite the fact that the models are latent diffusion.

![r2](https://github.com/lllyasviel/IC-Light/assets/19834515/25068f6a-f945-4929-a3d6-e8a152472223)

From left to right are inputs, model outputs relighting, devided shadow image, and merged normal maps. Note that the model is not trained with any normal map data. This normal estimation comes from the consistency of relighting.

You can reproduce this experiment using this button (it is 4x slower because it relight image 4 times)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/d9c37bf7-2136-446c-a9a5-5a341e4906de)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/fcf5dd55-0309-4e8e-9721-d55931ea77f0)

Below are bigger images (feel free to try yourself to get more results!)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/12335218-186b-4c61-b43a-79aea9df8b21)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/2daab276-fdfa-4b0c-abcb-e591f575598a)

For reference, [geowizard](https://fuxiao0719.github.io/projects/geowizard/) (geowizard is a really great work!):

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/4ba1a96d-e218-42ab-83ae-a7918d56ee5f)

And, [switchlight](https://arxiv.org/pdf/2402.18848) (switchlight is another great work!):

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/fbdd961f-0b26-45d2-802e-ffd734affab8)

# Model Notes

* **iclight_sd15_fc.safetensors** - The default relighting model, conditioned on text and foreground. You can use initial latent to influence the relighting.

* **iclight_sd15_fcon.safetensors** - Same as "iclight_sd15_fc.safetensors" but trained with offset noise. Note that the default "iclight_sd15_fc.safetensors" outperform this model slightly in a user study. And this is the reason why the default model is the model without offset noise.

* **iclight_sd15_fbc.safetensors** - Relighting model conditioned with text, foreground, and background.

Also, note that the original [BRIA RMBG 1.4](https://huggingface.co/briaai/RMBG-1.4) is for non-commercial use. If you use IC-Light in commercial projects, replace it with other background replacer like [BiRefNet](https://github.com/ZhengPeng7/BiRefNet).

# Cite

    @Misc{iclight,
      author = {Lvmin Zhang and Anyi Rao and Maneesh Agrawala},
      title  = {IC-Light GitHub Page},
      year   = {2024},
    }

# Related Work

Also read ...

[Total Relighting: Learning to Relight Portraits for Background Replacement](https://augmentedperception.github.io/total_relighting/)

[Relightful Harmonization: Lighting-aware Portrait Background Replacement](https://arxiv.org/abs/2312.06886)

[SwitchLight: Co-design of Physics-driven Architecture and Pre-training Framework for Human Portrait Relighting](https://arxiv.org/pdf/2402.18848)
