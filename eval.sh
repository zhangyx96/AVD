source activate pysc2
CUDA_VISIBLE_DEVICES=6 python src/main.py --config=qmix --env-config=mpe with \
evaluate=True test_nepisode=10 save_replay=True \
checkpoint_path='/space1/zhangyx/pymarl/results/models/qmix__2020-03-12_13-40-00'


