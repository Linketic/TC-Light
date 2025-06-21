<p align="center">
<h1 align="center"><strong>TC-Light: Temporally Consistent Relighting for Dynamic Long Videos</strong></h1>
  <p align="center">
    <em>Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences</em>
  </p>
</p>

<div id="top" align="center">

[![](https://img.shields.io/badge/%F0%9F%9A%80%20-Project%20Page-blue)](https://dekuliutesla.github.io/tclight/)
[![](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/TeslaYang123/CityGaussian)
![GitHub Repo stars](https://img.shields.io/github/stars/DekuLiuTesla/CityGaussian)

</div>
<p align="center">
  <iframe src="//player.bilibili.com/player.html?isOutside=true&aid=114721730598474&bvid=BV1S5N1zME7W&cid=30623665622&p=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>
</p>

This repo contains official implementations of **TC-Light**, a one-shot model used to manipulate the illumination of **high-dynamic videos** such as motion-rich actions and frequent switch of foreground and background objects. It is distinguished by:

- 🔥 Outstanding Temporal Consistency on Highly Dynamic Scenarios.
- 🔥 Superior Computational Efficiency that Enables Long Video Processing (can process 300 frames with resolution of 1280x720 on 40G A100).

These features make it particularly suitable for sim2real and real2real augmentation for Embodied Agents or preparing video pairs to train stronger video relighting models. Star ⭐ us if you like it!


## ⚡ Quick Start

Install the required environment as follows:
```bash
git clone https://github.com/Linketic/TC-Light.git
cd TC-Light
conda create -n tclight python=3.10
conda activate tclight
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
Then download required model weights to `./models` from the following links:

- **Hugging Face**: https://huggingface.co/TeslaYang123/TC-Light
- **Baidu Netdisk**: https://pan.baidu.com/s/1L-mk6Ilzd2o7KLAc7-gIHQ?pwd=rj99


As a quick start, you can use:
```bash
# support .mp4, .gif, .avi, and folder containing sequential images
# --multi_axis enables decayed multi-axis denoising, which enhances consistency but slow down the diffusion process
python run.py -i /path/to/your/video -p "your_prompt" \
              -n "your_negative_prompt" \  #  optional
              --multi_axis  # optional
```
By default, it will relight the first 30 frames with resolution 960x720. The default negative prompt is adopted from [Cosmos-Transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1), which makes the edited illumination as real as possible. If it is the first-time running on a specific video, it would generate and save flow un the path to your video. 

For a fine-grained control, you can customize your .yaml config file and run:
```bash
python run.py --config path/to/your_config.yaml
```
You can start from [configs/tclight_custom.yaml](configs/tclight_custom.yaml), which records the most frequently used parameters and detailed explanation.

<details>
<summary><span style="font-weight: bold;">Examples</span></summary>

  #### relight `examples/droid.mp4`
  ```bash
  python run.py --config configs/examples/tclight_droid.yaml
  ```

  #### relight `examples/navsim.mp4`
  ```bash
  python run.py --config configs/examples/tclight_navsim.yaml
  ```

  #### relight `examples/scand.avi`
  ```bash
  python run.py --config configs/examples/tclight_scand.yaml
  ```

  #### relight all three videos parallelly
  ```bash
  bash scripts/relight.sh
  ```
</details>


## 🔎 Behaviors
1. Works better on video resolution over 512x512, which is the minimum resolution used to train IC-Light. 
2. Works relatively better on realistic scenes than synthetics scenes, no matter in temporal consistency or instruction following ability.
3. Stuggle to drastically change illumination of night scenarios or hard shadows, as done in IC-Light.

## 📝 TODO List
- [x] Release the arXiv and the project page.
- [x] Release the code base.
- [ ] Release the dataset.

## Model Notes

* **iclight_sd15_fc.safetensors** - The default relighting model, conditioned on text and foreground. You can use initial latent to influence the relighting.

* **iclight_sd15_fcon.safetensors** - Same as "iclight_sd15_fc.safetensors" but trained with offset noise. Note that the default "iclight_sd15_fc.safetensors" outperform this model slightly in a user study. And this is the reason why the default model is the model without offset noise.

* **iclight_sd15_fbc.safetensors** - Relighting model conditioned with text, foreground, and background.

Also, note that the original [BRIA RMBG 1.4](https://huggingface.co/briaai/RMBG-1.4) is for non-commercial use. If you use IC-Light in commercial projects, replace it with other background replacer like [BiRefNet](https://github.com/ZhengPeng7/BiRefNet).

## 🤗 Citation
If you find this repository useful for your research, please use the following BibTeX entry for citation.

    @Misc{tclight,
      author = {Yang Liu, Chuanchen Luo, Zimo Tang, Yingyan Li, Yuran Yang, Yuanyong Ning, Lue Fan, Junran Peng, Zhaoxiang Zhang},
      title  = {TC-Light GitHub Page},
      year   = {2025},
    }

## 👏 Acknowledgements

This repo benefits from [IC-Light](https://github.com/lllyasviel/IC-Light/), [VidToMe](https://github.com/lixirui142/VidToMe/), [Slicedit](https://github.com/fallenshock/Slicedit/), [RAVE](https://github.com/RehgLab/RAVE), [Cosmos](https://github.com/NVIDIA/Cosmos). Thanks for their great work!

## ❓ FAQ
- _Out of memory occurs in training._ To finish training with limited VRAM, downsampling images or adjusting max_cache_num (we used a rather large 1024) in train_large.py can be a useful practice. Besides, you can increase `prune_ratio` in parallel tuning to further reduce memory cost.