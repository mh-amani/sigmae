# make sure to activate the correct conda environment
source /dlabscratch1/amani/miniconda3/bin/activate sigmae

cd /dlabscratch1/amani/sigmae
echo "Starting training"
pwd

# # supervised scan to 100% accuracy
# python ./src/train.py \
#     experiment=scan_bart \
#     trainer.min_epochs=50 \
#     trainer.max_epochs=100 \
#     data.supervision_ratio=[0.99,0.99] \
#     model.optimizer.lr=0.0001 \
#     data.batch_size=512

# unsupervised scan zxz to 100% accuracy
# python ./src/train.py \
#     experiment=scan_bart \
#     model.models_config.discretizer_x.config.quantize_vector_prob=1.0 \
#     model.models_config.discretizer_z.config.quantize_vector_prob=1.0 \
#     model/components/discretizers@model.models_config.discretizer_x=softmaxDB \
#     model/components/discretizers@model.models_config.discretizer_z=softmaxDB \
#     trainer.min_epochs=500 \
#     trainer.max_epochs=10000 \
#     data.supervision_ratio=[0.0,1.0] \
#     model.optimizer.lr=0.0001 \
#     data.batch_size=256


# # unsupervised image zxz mnist cnn
# python ./src/train.py \
#     experiment=mnist_vit_bart \
#     model/components/discretizers@model.models_config.discretizer_x=gumbelDB \
#     model.models_config.discretizer_x.config.quantize_vector_prob=0.5 \
#     model.model_params.x_vocab_size=10 \
#     model.model_params.max_x_length=12 \
#     trainer.min_epochs=50 \
#     trainer.max_epochs=100 \
#     data.supervision_ratio=[0.0,1.0] \
#     model.optimizer.lr=0.00004 \
#     data.batch_size=128

# continue training for
python ./src/train.py --config-path=/dlabscratch1/amani/sigmae/logs/train/runs/2024-09-29_11-50-28/.hydra --config-name=config \
    ckpt_path="/dlabscratch1/amani/sigmae/logs/train/runs/2024-09-29_11-50-28/checkpoints/last.ckpt" \
    model.optimizer.lr=0.00003 model.scheduler.factor=0.95 model.scheduler.patience=1 model.scheduler.threshold=0.05 model.scheduler.cooldown=0 \
    trainer.max_epochs=500 trainer.min_epochs=500 model.scheduler.scheduler_config.monitor=learn/loss data.batch_size=280

    # data.dataset.dataset_name_or_path="/dlabscratch1/amani/sigmae/data/cifar10" \


# # # # unsupervised image zxz cifar cnn vision_transformer or vision_transformer_pretrained
# python ./src/train.py \
#     experiment=cifar_bart_cnn \
#     model/components/sequence_models@model.models_config.sequence_model_zx=vision_transformer \
#     model/components/discretizers@model.models_config.discretizer_x=gumbelDB \
#     model.models_config.discretizer_x.config.quantize_vector_prob=0.5 \
#     model.model_params.x_vocab_size=256 \
#     model.model_params.max_x_length=50 \
#     trainer.min_epochs=400 \
#     trainer.max_epochs=400 \
#     trainer.accumulate_grad_batches=1 \
#     data.supervision_ratio=[0.0,1.0] \
#     model.optimizer.lr=0.0001 \
#     data.batch_size=64

# # # unsupervised image zxz vqvae cifar
# python ./src/train.py \
#     experiment=cifar_vqvae_bart \
#     model.models_config.discretizer_x.config.quantize_vector_prob=0.6 \
#     model/components/discretizers@model.models_config.discretizer_x=gumbelDB \
#     model/components/discretizers@model.models_config.discretizer_z=gumbelDB \
#     model.model_params.z_vocab_size=100 \
#     model.model_params.max_z_length=20 \
#     trainer.min_epochs=100 \
#     trainer.max_epochs=200 \
#     trainer.accumulate_grad_batches=1 \
#     data.supervision_ratio=[0.0,1.0] \
#     model.optimizer.lr=0.0001 \
#     data.batch_size=42


# # # unsupervised text2text on vqvae tokens
# python ./src/train.py \
#     experiment=cifartokonly_vqvae_bart \
#     model.models_config.discretizer_x.config.quantize_vector_prob=0.6 \
#     model.models_config.discretizer_z.config.quantize_vector_prob=0.6 \
#     model/components/discretizers@model.models_config.discretizer_x=gumbelDB \
#     model/components/discretizers@model.models_config.discretizer_z=gumbelDB \
#     trainer.min_epochs=100 \
#     trainer.max_epochs=200 \
#     trainer.accumulate_grad_batches=1 \
#     data.batch_size=64 \
#     data.supervision_ratio=[0.05,1.0] \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 \
#     callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 \
#     callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
#     model.optimizer.lr=0.0001
    


# # unsupervised image zxz imagenette cnn
# python ./src/train.py \
#     experiment=imagenette_bart_cnn \
#     model/components/discretizers@model.models_config.discretizer_x=gumbelDB \
#     model.models_config.discretizer_x.config.quantize_vector=True \
#     model.model_params.x_vocab_size=100 \
#     model.model_params.max_x_length=12 \
#     trainer.min_epochs=100 \
#     trainer.max_epochs=200 \
#     data.supervision_ratio=[0.0,1.0] \
#     model.optimizer.lr=0.00008 \
#     data.batch_size=32


