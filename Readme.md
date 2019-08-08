# SVM Classification

ML pipeline
```
# 1
python src/prepare.py data/iris.csv data/prepared/
dvc run -f prepare.dvc -d src/prepare.py -d data/iris.csv -o data/prepared/ python src/prepare.py data/iris.csv data/prepared/

# 2
python src/trainModel.py data/prepared result/svmmodel.pkl
dvc run -f train.dvc -d src/trainModel.py -d data/prepared -o result/svmmodel.pkl python src/trainModel.py data/prepared result/svmmodel.pkl

# 3
python src/evaluate.py data/prepared result/svmmodel.pkl acc.metric
dvc run -f evaluate.dvc -d src/evaluate.py -d data/prepared -d result/svmmodel.pkl -M acc.metric python src/evaluate.py data/prepared result/svmmodel.pkl acc.metric
```
