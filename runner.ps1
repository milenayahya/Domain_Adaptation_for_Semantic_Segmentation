conda activate aml
# echo "Running AML - TRAIN"
# python -m train.train
# echo "Running AML - TRAIN ADA"
# python -m train.trainADA
echo "Running AML - TRAIN FDA"
python -m train.train_FDA
echo "DONE"