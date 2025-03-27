
from plugin.VidToMe.utils import load_config, get_frame_ids, seed_everything

from invert import Inverter
from generate import Generator

from utils.model_utils import init_iclight

if __name__ == "__main__":
    config = load_config()
    # pipe, scheduler, model_key = init_model(
    #     config.device, config.sd_version, config.model_key, config.generation.control, config.float_precision)
    # manually change the pipe and scheduler
    pipe, scheduler, config.model_key = init_iclight(config.device)
    seed_everything(config.seed)

    # inversion = Inverter(vae, pipe, dpmpp_2m_sde_karras_scheduler_inv, config)
    # inversion(config.input_path, config.inversion.save_path)

    generator = Generator(pipe, scheduler, config)

    frame_ids = get_frame_ids(
        config.generation.frame_range, generator.data_parser.n_frames, config.generation.frame_ids)
    config.total_number_of_frames = len(frame_ids)

    generator(config.input_path, config.generation.latents_path,
              config.generation.output_path, frame_ids=frame_ids)
