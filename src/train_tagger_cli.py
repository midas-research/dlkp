from dlkp.models import KeyphraseTagger

if __name__ == "__main__":
    KeyphraseTagger.train_and_eval_cli()

# CUDA_VISIBLE_DEVICES=1 python train_tagger_cli.py --model_name_or_path bert-base-uncased --dataset_name "midas/inspec" --do_train --do_eval --pad_to_max_length --num_train_epochs 2 --output_dir /media/nas_mount/Debanjan/amardeep/dlkp_out/inpec_tagger_robertaa_cli --per_device_train_batch_size 4 --label_all_tokens
