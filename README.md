# Multi-genre Classification on Literary Books
## Advanced Topics in Machine Learning
## AUTH, Data and Web Science Msc program

This is the repo for the "Advanced Topics in Machine Learning" course

This project concerns Multi-Label genre classification from book descriptions using various Multi-Label Learning techniques such as:
- One Vs Rest
- Classifier Chains
- Random k-Labelsets
- Deep Learning

Futhermore, the problem of class imbalance is adressed using different methods. These are: 
- Easy Ensemble
- SMOTE
- ADASYN 
- Text Augmentation.

Finally, we explore multiple Active Learning approaches to simulate a real world NLP problem where labeled data are often not available and their manual annotation is difficult and time consuming. The techniques we studied are: 
- Query By Committee
- Ranked batch-mode Sampling 
- Information Density.

For more information you can check our Report and Presentation files above.

### Dependencies
In order to reproduce this project it is highly recommended to create a new Python 3 virtual env and install packages from requirements.txt file
```sh
python3 -m venv advanced_ml_venv
source advanced_ml_venv/bin/activate
pip install -r requirements.txt
```
