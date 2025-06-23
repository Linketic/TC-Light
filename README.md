<p align="center">
<h1 align="center"><strong>TC-Light: Temporally Consistent Relighting for Dynamic Long Videos</strong></h1>
  <p align="center">
    <em>Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences</em>
  </p>
</p>

<div id="top" align="center">

[![](https://img.shields.io/badge/%F0%9F%9A%80%20-Project%20Page-blue)](https://dekuliutesla.github.io/tclight/)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)
![GitHub Repo stars](https://img.shields.io/github/stars/Linketic/TC-Light)

</div>

https://github.com/user-attachments/assets/9d09cdbf-72dc-42ac-8104-1d40cab1551f

This repo contains official implementations of **TC-Light**, a one-shot model used to manipulate the illumination of **high-dynamic videos** such as motion-rich actions and frequent switch of foreground and background objects. It is distinguished by:

- 🔥 Outstanding Temporal Consistency on Highly Dynamic Scenarios.
- 🔥 Superior Computational Efficiency that Enables Long Video Processing (can process 300 frames with resolution of 1280x720 on 40G A100).

These features make it particularly suitable for sim2real and real2real augmentation for Embodied Agents or preparing video pairs to train stronger video relighting models. Star ⭐ us if you like it!

## 💡 Method

<div align="center">
    <img src='assets/pipeline.png'/>
</div>

**TC-Light** overview. Given the source video and text prompt p, the model tokenizes input latents in xy plane and yt plane seperately. The predicted noises are combined together for denoising. Its output then undergoes two-stage optimization to enhance temporal consistency of illumination and texture. Please refer to the paper for more details.

## 💾 Preparation

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

## ⚡ Quick Start

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

  #### relight the entire field of view
  ```bash
  python run.py --config configs/examples/tclight_droid.yaml
  ```
  ```bash
  python run.py --config configs/examples/tclight_navsim.yaml
  ```
  ```bash
  python run.py --config configs/examples/tclight_scand.yaml
  ```

  #### relight all three videos parallelly
  ```bash
  bash scripts/relight.sh
  ```

  #### relight foreground with static background condition
  ```bash
  # we generate compatible background image by using foreground mode of IC-Light, then remove foreground and inpaint the image with tools like sider.ai
  # for satisfactory results, a consistent and complete foreground segmentation is preferred.
  python run.py --config configs/examples/tclight_bkgd_robotwin.yaml
  ```
</details>

For evaluation, you can simply use:
```bash
python evaluate.py --output_dir path/to/your_output_dir --eval_cost
```

## 🔎 Behaviors
1. Works better on video resolution over 512x512, which is the minimum resolution used to train IC-Light. A higher resolution helps consistency of image intrinsic properties.
2. Works relatively better on realistic scenes than synthetics scenes, no matter in temporal consistency or physical plausibility.
3. Stuggle to drastically change illumination of night scenarios or hard shadows, as done in IC-Light.

## 📝 TODO List
- [x] Release the arXiv and the project page.
- [x] Release the code base.
- [ ] Release the dataset.

## 🤗 Citation
If you find this repository useful for your research, please use the following BibTeX entry for citation.

    @Misc{tclight,
      author = {Yang Liu, Chuanchen Luo, Zimo Tang, Yingyan Li, Yuran Yang, Yuanyong Ning, Lue Fan, Junran Peng, Zhaoxiang Zhang},
      title  = {TC-Light GitHub Page},
      year   = {2025},
    }

## 👏 Acknowledgements

This repo benefits from [IC-Light](https://github.com/lllyasviel/IC-Light/), [VidToMe](https://github.com/lixirui142/VidToMe/), [Slicedit](https://github.com/fallenshock/Slicedit/), [RAVE](https://github.com/RehgLab/RAVE), [Cosmos](https://github.com/NVIDIA/Cosmos). Thanks for their great work! The repo is still under development, we are open to pull request and discussions!

