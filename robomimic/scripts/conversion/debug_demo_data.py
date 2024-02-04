import h5py

def read_actions_states(file_path):
    with h5py.File(file_path, 'r') as file:
        print(type(file))
        print(file.keys())
        data_group = file['data']
        print(data_group.keys())
        for demo_name, demo_group in data_group.items():
            print(f"Reading demonstration: {demo_name}")
            print(demo_group.keys())

            import pdb; pdb.set_trace()


# read_actions_states("/home/dpenmets/LIRA_work/robomimic_vd/robomimic/scripts/final_cfgs/demo_data/28_JAN/28_27_combined.hdf5")
# read_actions_states("/home/dpenmets/LIRA_work/robomimic_vd/robomimic/scripts/final_cfgs/demo_data/28_JAN/28_27_combined_demos.hdf5")
# read_actions_states("/home/dpenmets/LIRA_work/robomimic_vd/robomimic/scripts/final_cfgs/demo_data_with_gaze/low_dim.hdf5")
read_actions_states("/home/dpenmets/LIRA_work/robomimic_vd/robomimic/scripts/final_cfgs/demo_data_with_gaze/demo.hdf5")