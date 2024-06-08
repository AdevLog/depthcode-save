# depthcode-save
Glass Detection for Depth Completion Lite ver


Data Source: https://github.com/tsunghan-wu/Depth-Completion
> --> Data_140 include GT Data and Render Data, total 140 pics.\
--> no_glass_275 is non-glass dataset.\
--> txt file include raw depth file paths, separate to GT Data and Render Data, modify it as needed.

--------
Preprocessing:
matterport3D's data include noise, require to execute clean_depth_noise.py first.

find glass area(output: uint8): glasscut.py\
depth completeion method: glass_depth_completion.py
> --> input file path for raw depth, reprocessing depth, mask.\
	--> input file path for saving complete depth files.

glass area metrics: eval_glass_mask_metrics.py\
depth completion metrics: eval_depth_completion_metrics.py


Ablation study in different methods: \
OpenCV inpainting: cv_inpainting.py\
Scipy griddata: griddata.py

npy file to exr file: BDepth_npy_to_exr.py\
exr file to png file: convert_exr_depth_to_png.py

gray png file to rgb color: show_rgb_depth.py
