CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
--master_port=58198 \
main.py --do_train \
--num_thread_reader=4 \
--data_path /home/zyl/MeVTR_data_and_models/charades/annotation \
--features_path /home/zyl/MeVTR_data_and_models/charades/Charades_v2_3 \
--output_dir /home/zyl/flexible_k/output_shuffle_lstm_test \
--max_words 77 --max_frames 64 --batch_size_val 16 \
--datatype charades --feature_framerate 1 --coef_lr 1e-4 \
--slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqLSTM \
--pretrained_clip_name ViT-B/32 \
--post_process none --post_cluster_centroids 16 \
--batch_size 16 --shuffle_events \
--shuffle_exp