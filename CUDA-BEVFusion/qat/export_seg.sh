python qat/export-camera_seg.py --ckpt=./model/seg_camera_only_resnet50/seg_camera_only_resnet50_fp16.pth --fp16
# python qat/export-transfuser_seg.py --ckpt=./model/seg_camera_only_resnet50/seg_camera_only_resnet50_fp16.pth --fp16
# python qat/export-scn.py --ckpt=./qat/ckpt/bevfusion_fp16.pth --save=qat/onnx_fp16/lidar.backbone.onnx