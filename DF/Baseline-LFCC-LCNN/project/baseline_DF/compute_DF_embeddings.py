#!/usr/bin/env python
"""
main.py

The default training/inference process wrapper
Requires model.py and config.py

Usage: $: python main.py [options]
"""
from __future__ import absolute_import
import glob
import numpy as np
import os
import soundfile as sf
import sys
import torch
import importlib

import core_scripts.other_tools.display as nii_warn
import core_scripts.data_io.default_data_io as nii_dset
import core_scripts.data_io.conf as nii_dconf
import core_scripts.other_tools.list_tools as nii_list_tool
import core_scripts.config_parse.config_parse as nii_config_parse
import core_scripts.config_parse.arg_parse as nii_arg_parse
import core_scripts.op_manager.op_manager as nii_op_wrapper
import core_scripts.nn_manager.nn_manager as nii_nn_wrapper
import core_scripts.startup_config as nii_startup
import core_scripts.other_tools.display as nii_display
import core_scripts.nn_manager.nn_manager_tools as nii_nn_tools
import core_scripts.nn_manager.nn_manager_conf as nii_nn_manage_conf


__author__ = "Xin Wang, Xuechen Liu"
__email__ = "wangxin@nii.ac.jp, xuechen.liu@uef.fi"
__copyright__ = "Copyright 2022, Xin Wang & Xuechen Liu"


def main():
    """main(): the default wrapper for training and inference process
    Please prepare config.py and model.py
    """
    # arguments initialization
    args = nii_arg_parse.f_args_parsed()

    #
    nii_warn.f_print_w_date("Start program", level="h")
    nii_warn.f_print("Load module: %s" % (args.module_config))
    nii_warn.f_print("Load module: %s" % (args.module_model))
    prj_conf = importlib.import_module(args.module_config)
    prj_model = importlib.import_module(args.module_model)

    # initialization
    nii_startup.set_random_seed(args.seed, args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # prepare data io

    # default, no truncating, no shuffling
    params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
    }

    if type(prj_conf.test_list) is list:
        t_lst = prj_conf.test_list
    else:
        t_lst = nii_list_tool.read_list_from_text(prj_conf.test_list)
    test_set = nii_dset.NIIDataSetLoader(
        prj_conf.test_set_name,
        t_lst,
        prj_conf.test_input_dirs,
        prj_conf.input_exts,
        prj_conf.input_dims,
        prj_conf.input_reso,
        prj_conf.input_norm,
        prj_conf.test_output_dirs,
        prj_conf.output_exts,
        prj_conf.output_dims,
        prj_conf.output_reso,
        prj_conf.output_norm,
        "./",
        params=params,
        truncate_seq=None,
        min_seq_len=None,
        save_mean_std=False,
        wav_samp_rate=prj_conf.wav_samp_rate,
        global_arg=args,
    )

    # initialize model
    model = prj_model.Model(
        test_set.get_in_dim(), test_set.get_out_dim(), args, prj_conf
    )
    if args.trained_model == "":
        print("No model is loaded by ---trained-model for inference")
        print("By default, load %s%s" % (args.save_trained_name, args.save_model_ext))
        checkpoint = torch.load("%s%s" % (args.save_trained_name, args.save_model_ext))
    else:
        checkpoint = torch.load(args.trained_model)

    # cuda device
    if torch.cuda.device_count() > 1 and args.multi_gpu_data_parallel:
        nii_display.f_print("DataParallel for inference is not implemented", "warning")
    nii_display.f_print("\nUse single GPU: %s\n" % (torch.cuda.get_device_name(device)))

    # print the network
    model.to(device, dtype=nii_dconf.d_dtype)
    nii_nn_tools.f_model_show(model)

    # load trained model parameters from checkpoint
    cp_names = nii_nn_manage_conf.CheckPointKey()
    if type(checkpoint) is dict and cp_names.state_dict in checkpoint:
        model.load_state_dict(checkpoint[cp_names.state_dict])
    else:
        model.load_state_dict(checkpoint)

    # # do inference and output data
    # nii_nn_wrapper.f_inference_wrapper(args, model, device, test_set, checkpoint)
    # # done

    srcdir = sys.argv[1]
    tardir = sys.argv[2]
    os.makedirs(tardir + "/npys", exist_ok=True)

    flac_files = glob.glob(srcdir + "/*.flac")
    print("{0} files are gonna be processed".format(len(flac_files)))

    with torch.no_grad():
        for flac_path in flac_files:
            x, _ = sf.read(flac_path)
            utt = os.path.basename(flac_path).split(".")[0]
            embed_x = model.compute_embedding(x)
            embed_x = embed_x.cpu().numpy()
            np.save(tardir + "/npys/{0}.npy".format(utt), embed_x)


if __name__ == "__main__":
    main()
