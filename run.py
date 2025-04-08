
from plugin.VidToMe.utils import load_config, get_frame_ids, seed_everything, init_model

from invert import Inverter
from generate import Generator

from utils.model_utils import init_iclight

if __name__ == "__main__":
    config = load_config()
    seed_everything(config.seed)

    if config.sd_version == 'iclight':
        pipe, scheduler, config.model_key = init_iclight(config.device)
        
        config.max_memory_allocated = 0
        config.total_time = 0
    else:
        pipe, scheduler, config.model_key = init_model(
            config.device, config.sd_version, config.model_key, config.generation.control, config.float_precision)
        inversion = Inverter(pipe, scheduler, config)
        print("Start inversion!")
        inversion(config.inversion.save_path)

        config.max_memory_allocated = inversion.config.max_memory_allocated
        config.total_time = inversion.config.total_time

    generator = Generator(pipe, scheduler, config)

    frame_ids = get_frame_ids(
        config.generation.frame_range, generator.data_parser.n_frames, config.generation.frame_ids)
    config.total_number_of_frames = len(frame_ids)

    generator(config.generation.latents_path, config.generation.output_path, frame_ids=frame_ids)
