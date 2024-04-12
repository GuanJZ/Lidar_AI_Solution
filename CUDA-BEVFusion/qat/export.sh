python qat/export-camera.py --ckpt=./qat/ckpt/bevfusion_fp16.pth --fp16
python qat/export-transfuser.py --ckpt=./qat/ckpt/bevfusion_fp16.pth --fp16
python qat/export-scn.py --ckpt=./qat/ckpt/bevfusion_fp16.pth --save=qat/onnx_fp16/lidar.backbone.onnx