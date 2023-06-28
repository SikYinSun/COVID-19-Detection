# <b>  COMPARISON OF DISTINCT CNN BACKBONE ARCHITECTURES WITH DENSE HEADS, ATTENTION AND VISUAL TRANSFORMERS ON MEDICAL X-RAY IMAGE DATASET  </b>
# AASD 4015 - Advance Mathematical Concepts for Deep Learning Group Project 2

## Deploy website: [https://sikyinsun.github.io/AASD-4015---COVID-19-Detection/](https://sikyinsun.github.io/COVID-19-Detection/)

<b>Members:</b> 
1. Saksham Prakash (101410709) 
2. Sik Yin Sun (101409665)

## Background and Motivation

The COVID-19 pandemic has had a profound impact on global health and early diagnosis of COVID-19 is critical for effective management of the disease.

For this project, we explore <b> different backbone architecures </b> and their results on a binary-classification task using images of X-Rays of patients with/without the disease. 
Additionally, since the <b> attention mechanism </b> has been quite successful recently, our attempts on utilizing the same for our purpose, has been detailed as well.

---

In the context of COVID screening from X-rays, a <b> high recall would be preferred over high precision </b>. This is because the consequences of a false negative - i.e., failing to identify a COVID-19 infection in a patient who actually has the disease - can be severe. Patients with COVID-19 can develop severe symptoms and require urgent medical attention, and failure to detect the disease can result in delayed treatment and increased transmission.

On the other hand, the consequences of a false positive - i.e., identifying a patient as having COVID-19 when they do not - are less severe. In the case of false positives, the patient may be required to undergo additional testing or quarantine measures, but this is a relatively minor inconvenience compared to the potential harm of a false negative.

Therefore, in the case of COVID screening from X-rays, a high recall model that detects as many true positive cases as possible, even if it results in some false positives, is preferred over a high precision model that minimizes false positives at the expense of potentially missing true positive cases.

---

## Problem Statement
Given a set of images of chest X-Rays, classify whether the patients are infected with Covid or Normal.

## Contents
| SNo. | Contents
| -------- | -------- 
| 1 | <b> Libraries Setup </b> 
| 2 | <b> Data - Chest X-Rays </b>
| 2.1 | Download 
| 2.2 | Initial Exploration 
| 2.3 | Handling Class Imbalance by Oversampling
| 3 | <b> Helper Functions to Evaluate Model Trainings Uniformally </b>
| 4 | <b> Training baseline backbone architectures with simple head </b>
| 4.1 | VGG-16
| 4.2 | InceptionV3 
| 4.3 | EffientNetV2B3  
| 5 | <b> VGG-16 backbone with Attention heads </b>
| 5.1 | VGG-16 backbone with Self-Attention 
| 5.2 | VGG-16 backbone with Multi-Head Attention 
| 6 | <b> Visual Transformers (ViT) </b>
| 7 | <b> Results </b>
| 8 | <b> Conclusion </b>
| 9 | <b> Learning Outcomes </b> 
| 10 | <b> References </b> 

## <b> Results </b>
<a id='results'></a>

Precision and Recall of Pneumonia class were used.

| SNo. | Training | Parameters | Test Accuracy | Precision | Recall | F1 Score | Comments
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 1 | VGG-16 + Dense head | 27,822,401 | 0.6872 | 0.5  | <b> 0.82 </b>| <b> 0.62 </b> | Highest Recall, F1
| 2 | Inception-V3 + Dense head | 55,357,729 | 0.7218 | 0.51 | 0.78 | 0.62 |
| 3 | EfficientNetV2B3 + Dense head | 52,252,735 | 0.5000 | 1.00 | 0.00 | 0.00 | 
| 4 | VGG-16 + SelfAttention Head | 14,813,057 | 0.7692 | 0.50 | 0.69 | 0.58 | Smallest model
| 5 | VGG-16 + MultiHead Attention | 34,375,617 | <b> 0.8128 </b> | 0.50 | 0.67 | 0.57 | Highest Test Accuracy
| 6 | ViT | 88,085,761 | 0.5000 | 1.00 | 0.00 | 0.00 | 

## <b> Conclusions </b>
<a id='conclusion'></a>

In this project, we explored different backbone architectures along with dense heads, attention heads, and visual transformers for the task of binary classification of chest X-ray images to detect COVID-19. Our goal was to find a model that could achieve a high recall value for the "PNEUMONIA" class, as the consequences of a false negative can be severe in the context of COVID-19 diagnosis.

From our experiments, we observed that the VGG-16 with a Dense head provided the highest recall and F1 score, making it the most effective model for our purpose. Furthermore, the VGG-16 with Multi-Head Attention achieved the highest test accuracy among all the models. On the other hand, models based on EfficientNetV2B3 and ViT did not perform well on this specific task.

It is important to note that the results may vary depending on the choice of training parameters, dataset, and the specific problem under consideration. In the future, we could explore other backbone architectures, attention mechanisms, or even ensemble methods to further improve the model's performance. Additionally, techniques such as data augmentation and fine-tuning of model parameters could be employed to enhance

## <b> Learning Outcomes </b>
<a id='learningOuts'></a>

1. Gained experience in implementing custom models in Keras with custom architectures such as self-attention and multi-head attention mechanisms.
   
2. Explored various backbone architectures, including VGG-16, InceptionV3, EfficientNetV2B3, and Visual Transformers (ViT), for the medical image classification task.
   
3. Understood the importance of recall in the context of COVID-19 diagnosis from chest X-ray images and the potential consequences associated with false negatives.
   
4. Learned how to handle class imbalance in the dataset using oversampling techniques.
   
5. Developed skills in evaluating model performance using metrics such as accuracy, precision, recall, F1 score, confusion matrix, and classification reports.
   
6. Gained insights on the trade-offs between different models in terms of performance, model complexity, and size.
   
7. Understood the potential areas for improvement and future work, such as exploring other backbone architectures, ensemble methods, data augmentation, and fine-tuning of model parameters.

## <b> References </b>
<a id='ref'></a>


1. Mooney, P. T. (n.d.). Chest X-Ray Images (Pneumonia). Kaggle. Retrieved from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
   
2. Lemaitre, G., Nogueira, F., & Aridas, C. K. (n.d.). imblearn.over_sampling.RandomOverSampler. imbalanced-learn. Retrieved from https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
   
3. Weights & Biases Authors. (n.d.). Simple Ways to Tackle Class Imbalance. Weights & Biases. Retrieved from https://wandb.ai/authors/class-imbalance/reports/Simple-Ways-to-Tackle-Class-Imbalance--VmlldzoxODA3NTk
   
4. Stack Overflow Contributors. (2017, November 14). Oversampling functionality in TensorFlow Dataset API. Stack Overflow. Retrieved from https://stackoverflow.com/questions/47236465/oversampling-functionality-in-tensorflow-dataset-api
   
5. Mooney, P. T. (n.d.). Exploring the Kaggle API. Kaggle. Retrieved from https://www.kaggle.com/code/paultimothymooney/exploring-the-kaggle-api
   
6. Morales, F. (n.d.). vit-keras: Implementation of Vision Transformers in Keras. GitHub. Retrieved from https://github.com/faustomorales/vit-keras
   
7. Morales, F., & Contributors (n.d.). Issue #35: How to use ViT for regression? GitHub. Retrieved from https://github.com/faustomorales/vit-keras/issues/35
   
8. Papers with Code Contributors (n.d.). Vision Transformer Method Page on Papers with Code. Papers with Code. Retrieved from https://paperswithcode.com/method/vision-transformer
   
9.  TensorFlow Contributors (n.d.). tf.keras.layers.MultiHeadAttention API documentation page on TensorFlow.org. TensorFlow.org. Retrieved from https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention 
    
10. TensorFlow Contributors (n.d.). Neural machine translation with attention tutorial on TensorFlow.org Text Tutorials section.TensorFlow.org.Retrieved from https://www.tensorflow.org/text/tutorials/nmt_with_attention
    
11. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser,L., & Polosukhin,I.(2017). Attention Is All You Need.arXiv preprint arXiv:1706.03762.Retrieved from https://arxiv.org/pdf/1706.03762.pdf
    
12. ChatGPT. (2023, April 1 - 2023, April 6). [Mostly debugging questions from code implementation of concepts] [Response to the user questions]. Retrieved from https://chat.openai.com/chat
