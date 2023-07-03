import argparse
import os
import os.path as path

from runner import get_runner
from utils import get_config, save_config, setup_logger

if __name__ == "__main__":
    # get configuration from command line
    parser = argparse.ArgumentParser(
        description="Running a skeleton prediction network.")
    parser.add_argument("--exp_name",
                        default="test_model",
                        type=str,
                        help="experiment names")
    parser.add_argument("--run_dir",
                        default="run/",
                        type=str,
                        help="result/dir")
    parser.add_argument("--config",
                        default="configs/config.yaml",
                        help="default configs")
    args = parser.parse_args()

    # get configuration from file
    opts = get_config(args.config)
    opts["save"]["path"]["base"] = args.run_dir
    if not path.isdir(args.run_dir):
        os.makedirs(args.run_dir)

    logger = setup_logger("prediction", args.run_dir, 0)
    logger.info(f"Pid: {os.getpid()}")

    if "test" not in opts["mode"]:  # only training options are saved
        save_config(opts, path.join(args.run_dir, "train_options.yaml"), True,
                    logger)

    opts["logger"] = logger
    runner = get_runner(opts["runner"], opts)
    runner.run()
