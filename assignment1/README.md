# Assignment 1: Linear Classifiers

In this assigment, we implement various linear classifiers, namely logistic regression, perceptron, supported vector machine, and softmax, and train them on the [Rice dataset](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification) and the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

## Model
We implement the following classifiers (in their respective files):
1. Logistic regression (`logistic.py`)
2. Perceptron (`perceptron.py`)
3. SVM (`svm.py`)
4. Softmax (`softmax.py`)

For the logistic regression classifier, multi-class prediction is difficult, as it requires a one-vs-one or one-vs-rest classifier for every class. Therefore, we only use logistic regression on the Rice dataset.

## Data Setup
To set up dataset
```bash
$ cd assignment1/fashion-mnist/
$ bash get_data.sh
```

## Result
### Rice Dataset
| Benchmark | Test Acc. |
|---------|-----------|
| Logistic Regression | 99.23% | 
| Perceptron | 100% |
| SVM | 100% |
| Softmax | 100% |

### Fashion-MNIST Dataset
| Setting | Test Acc. |
|---------|-----------|
| Logistic Regression | Not Applicable |
| Perceptron | 83.06% |
| SVM | 84.15% |
| Softmax | 82.90% |

