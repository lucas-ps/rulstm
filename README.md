python extract_rgb.py --dataset_path "/media/lucas/Linux SSD/50-Salads/" --dataset 50-Salads --generate_jpgs --extract_features
python extract_rgb.py --dataset_path "/media/lucas/Linux SSD/BreakfastII_15fps_qvga_sync/" --dataset Breakfast --generate_jpgs --extract_features

mkdir models/

python main.py train data/50-salads models/50-salads --modality rgb --task anticipation --sequence_completion
python main.py train data/50-salads models/50-salads --modality rgb --task anticipation
python main.py validate data/50-salads models/50-salads --modality rgb --task anticipation
python main.py test data/50-salads models/50-salads --modality rgb --task anticipation --json_directory jsons/50-salads

python main.py train data/breakfast models/breakfast --modality rgb --task anticipation --sequence_completion
python main.py train data/breakfast models/breakfast --modality rgb --task anticipation
python main.py validate data/breakfast models/breakfast --modality rgb --task anticipation
python main.py test data/breakfast models/breakfast --modality rgb --task anticipation --json_directory jsons/breakfast

python main.py train data/ek55 models/ek55 --modality rgb --task anticipation --sequence_completion
python main.py train data/ek55 models/ek55 --modality rgb --task anticipation --ek100
python main.py validate data/ek55 models/ek55 --modality rgb --task anticipation
python main.py test data/ek55 models/ek55 --modality rgb --task anticipation --json_directory jsons/ek55

python main.py train data/egtea models/egtea --modality rgb --task anticipation --sequence_completion
