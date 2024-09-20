# main

- src/main.py 파일을 돌리자
- --ratio_known_normal <= 0.6  --ratio_known_outlier <= 0.4 웬만하면 건들지 않기.
- ratio_known_normal, ratio_known_outlier 비율은 전체 데이터에서 라벨 0,1 비율과 같음. 합이 1이어야 함.
- 아래는 예시 파서, 다양하게 돌려서 성능을 높여보자

''' 
python main.py custom custom_mlp ../log/DeepSAD/AKI_test ../data/ --ratio_known_normal 0.6  --ratio_known_outlier 0.4 --ratio_pollution 0.0 --lr 0.001 --n_epochs 100 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.001 --ae_n_epochs 100 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 0  --known_outlier_class 1 --n_known_outlier_classes 1 --seed 0 
'''  
