import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from _base_data_preprocessor import *

def main(task_name, task_config, expert_data_num):
    output_path = f"./data/sim-{task_name}/{task_config}-{expert_data_num}"
    
    task_settings_path = Path(__file__).parent / './task_settings.json'
    if task_settings_path.exists():
        with open(task_settings_path, 'r') as f:
            task_settings = json.load(f)
    else:
        task_settings = {}
    
    camera_type = task_settings.get(task_name, {}).get('camera_type', 'head')
    if camera_type == 'all':
        visual_cameras = ['head', 'wrist']
    else:
        visual_cameras = [camera_type]
    tactile_cameras = ['left', 'right']
    downsample_factor = task_settings.get(task_name, {}).get('downsample_factor', 1)
 
    processor = BaseDataPreprocessor(task_name, task_config)
    metadata = processor.run(
        save_root_path=output_path,
        visual_cameras=visual_cameras,
        tactile_cameras=tactile_cameras,
        downsample_factor=downsample_factor,
        episode_num=expert_data_num,
        random_select=False,
    )

    SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"
    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
        "dataset_dir": metadata['dataset_dir'],
        "num_episodes": metadata['num_episodes'],
        "episode_len": metadata['episode_len'],
        "camera_names": metadata['camera_names'],
    }

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TacArena episodes for ACT training.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., insert_hole)",
    )
    parser.add_argument("task_config", type=str, help="Task config (e.g., demo)")
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process")

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    main(task_name, task_config, expert_data_num)