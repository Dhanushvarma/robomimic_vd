"""
A useful script for generating json files and shell scripts for conducting parameter scans.
The script takes a path to a base json file as an argument and a shell file name.
It generates a set of new json files in the same folder as the base json file, and 
a shell file script that contains commands to run for each experiment.

Instructions:

(1) Start with a base json that specifies a complete set of parameters for a single 
    run. This only needs to include parameters you want to sweep over, and parameters
    that are different from the defaults. You can set this file path by either
    passing it as an argument (e.g. --config /path/to/base.json) or by directly
    setting the config file in @make_generator. The new experiment jsons will be put
    into the same directory as the base json.

(2) Decide on what json parameters you would like to sweep over, and fill those in as 
    keys in @make_generator below, taking note of the hierarchical key
    formatting using "/" or ".". Fill in corresponding values for each - these will
    be used in creating the experiment names, and for determining the range
    of values to sweep. Parameters that should be sweeped together should
    be assigned the same group number.

(3) Set the output script name by either passing it as an argument (e.g. --script /path/to/script.sh)
    or by directly setting the script file in @make_generator. The script to run all experiments
    will be created at the specified path.

Args:
    config (str): path to a base config json file that will be modified to generate config jsons.
        The jsons will be generated in the same folder as this file.

    script (str): path to output script that contains commands to run the generated training runs

Example usage:

    # assumes that /tmp/gen_configs/base.json has already been created (see quickstart section of docs for an example)
    python hyperparam_helper.py --config /tmp/gen_configs/base.json --script /tmp/gen_configs/out.sh
"""
import argparse

import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils


def make_generator(config_file, script_file):
    """
    Implement this function to setup your own hyperparameter scan!
    """
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file, script_file=script_file
    )

    ## GROUP 0  - fixed stuff

    generator.add_param(
        key="experiment.save.every_n_epochs",
        name="", 
        group=0, 
        values=[250],
    )


    generator.add_param(
        key="experiment.rollout.rate",
        name="", 
        group=0, 
        values=[250],
    )

    ## END of GROUP 0 

    ##GROUP 1 of hyper paramas



    generator.add_param(
        key = "algo.subgoal_update_interval",
        name = "subgoal_update_interval",
        group=1,
        values=[10, 50, 100]
    )


    generator.add_param(
        key = "algo.planner.subgoal_horizon",
        name="subgoal_horizon", 
        group=1, 
        values=[10, 50, 100] 
    )

    generator.add_param(
        key = "algo.actor.rnn.horizon",
        name="rnn_horizon", 
        group=1, 
        values=[10, 50, 100]
    )

    ### END of GROUP 1


    generator.add_param(
        key = "algo.planner.vae.prior.learn",
        name="vae_prior_learn", 
        group=2, 
        values=[False, True, True, True], 
        value_names=["f", "t", "t", "t"],
    )

    generator.add_param(
        key = "algo.planner.vae.prior.use_gmm",
        name="use_gmm", 
        group=2, 
        values=[False, True, True, True], 
        value_names=["f", "t", "t", "t"],
    )


    generator.add_param(
        key = "algo.planner.vae.prior.gmm_num_modes",
        name="gmm_num_modes", 
        group=2, 
        values=[0, 2, 4, 6], 
        value_names=[0, 2, 4, 6],
    )


    

    return generator


def main(args):

    # make config generator
    generator = make_generator(config_file=args.config, script_file=args.script)

    # generate jsons and script
    generator.generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to base json config - will override any defaults.
    parser.add_argument(
        "--config",
        type=str,
        help="path to base config json that will be modified to generate jsons. The jsons will\
            be generated in the same folder as this file.",
    )

    # Script name to generate - will override any defaults
    parser.add_argument(
        "--script",
        type=str,
        help="path to output script that contains commands to run the generated training runs",
    )

    args = parser.parse_args()
    main(args)
