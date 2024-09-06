# make sure to activate the correct conda environment
source /dlabscratch1/amani/miniconda3/bin/activate sigmae

# supervised scan to 100% accuracy
# python ./src/train.py \
#     experiment=scan_bart \
#     trainer.min_epochs=50 \
#     trainer.max_epochs=100 \
#     data.supervision_ratio=[0.99,0.99] \
#     model.optimizer.lr=0.0001 \
#     data.batch_size=512

# unsupervised scan zxz to 100% accuracy
python ./src/train.py \
    experiment=scan_bart \
    trainer.min_epochs=50 \
    trainer.max_epochs=100 \
    data.supervision_ratio=[0.0,1.0] \
    model.optimizer.lr=0.0001 \
    data.batch_size=210