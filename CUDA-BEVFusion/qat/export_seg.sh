python qat/ptq_for_fp16.py --config bevfusion/configs/nuscenes/seg/camera-bev-resnet50.yaml --ckpt model/seg_camera_only_resnet50/seg_camera_only_resnet50.pth --save_path model/seg_camera_only_resnet50/seg_camera_only_resnet50_fp16.pth
python qat/export-camera_seg.py --ckpt=./model/seg_camera_only_resnet50/seg_camera_only_resnet50_fp16.pth --fp16
# python qat/export-transfuser_seg.py --ckpt=./model/seg_camera_only_resnet50/seg_camera_only_resnet50_fp16.pth --fp16
# python qat/export-scn.py --ckpt=./qat/ckpt/bevfusion_fp16.pth --save=qat/onnx_fp16/lidar.backbone.onnx

python qat/ptq_for_fp16.py --config bevfusion/configs/nuscenes/seg/camera-bev-resnet50.yaml --ckpt ./model/seg_camera_only_resnet50_ge_bev_output_scope_0.5/latest.pth
python qat/export-camera_seg.py --ckpt=./model/seg_camera_only_resnet50_ge_bev_output_scope_0.5/latest_fp16.pth --fp16
python qat/export-transfuser_seg.py --ckpt=./model/seg_camera_only_resnet50_ge_bev_output_scope_0.5/latest_fp16.pth --fp16