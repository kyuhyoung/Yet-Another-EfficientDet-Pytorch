#python main_new.py --mode="val" --threshold=0.65 --iou_threshold=0.5 --ios_threshold=0.7 --letterbox_type=top_left --is_mosaic 
#python main_new.py --mode="val" --threshold=0.65 --iou_threshold=0.5 --ios_threshold=0.7 --letterbox_type=top_left --is_mosaic --use_pytorch_batched_nms 
python main_new.py --mode="val" --threshold=0.65 --iou_threshold=0.5 --ios_threshold=0.7 --letterbox_type=center --is_mosaic
#python main_new.py --mode="val" --threshold=0.65 --iou_threshold=0.5 --ios_threshold=0.7 --letterbox_type=center --is_mosaic --use_pytorch_batched_nms
#python main_new.py --mode="val" --threshold=0.65 --iou_threshold=0.5 --letterbox_type=center
#python main_new.py --mode="val" --threshold=0.65 --iou_threshold=0.5 --letterbox_type=top_left
