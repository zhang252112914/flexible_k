CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
--master_port=54198 \
main.py --do_eval \
--num_thread_reader=2 \
--data_path /home/zyl/MeVTR_data_and_models/charades/annotation \
--features_path /home/zyl/MeVTR_data_and_models/charades/Charades_v2_3 \
--output_dir /home/zyl/flexible_k/eval/eval_topk_one \
--max_words 77 --max_frames 64 --batch_size_val 16 \
--datatype charades --feature_framerate 1 --coef_lr 1e-3 \
--slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32 \
--post_process topk --post_cluster_centroids 16 \
--batch_size 16 --onlyone \
--init_model /home/zyl/flexible_k/output_one/pytorch_model.bin.4