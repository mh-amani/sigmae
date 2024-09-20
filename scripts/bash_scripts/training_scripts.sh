# make sure to activate the correct conda environment
source /dlabscratch1/amani/miniconda3/bin/activate sigmae

cd /dlabscratch1/amani/sigmae
echo "Starting training"
pwd

# supervised scan to 100% accuracy
# python ./src/train.py \
#     experiment=scan_bart \
#     trainer.min_epochs=50 \
#     trainer.max_epochs=100 \
#     data.supervision_ratio=[0.99,0.99] \
#     model.optimizer.lr=0.0001 \
#     data.batch_size=512



# # unsupervised scan zxz to 100% accuracy
# python ./src/train.py \
#     experiment=scan_bart \
#     model/components/discretizers@model.models_config.discretizer_x=gumbelDB \
#     model/components/discretizers@model.models_config.discretizer_z=gumbelDB \
#     trainer.min_epochs=500 \
#     trainer.max_epochs=10000 \
#     data.supervision_ratio=[0.0,1.0] \
#     model.optimizer.lr=0.0001 \
#     data.batch_size=128


# unsupervised image zxz
python ./src/train.py \
    experiment=mnist_vit_bart \
    model/components/discretizers@model.models_config.discretizer_x=gumbelDB \
    model.models_config.discretizer_x.config.quantize_vector=False \
    model.model_params.x_vocab_size=50 \
    model.model_params.max_x_length=12 \
    trainer.min_epochs=50 \
    trainer.max_epochs=100 \
    data.supervision_ratio=[0.0,1.0] \
    model.optimizer.lr=0.00004 \
    data.batch_size=128