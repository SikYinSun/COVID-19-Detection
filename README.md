# <b>  COMPARING DISTINCT CNN BACKBONE ARCHITECTURES WITH DENSE HEADS, ATTENTION AND VISUAL TRANSFORMERS ON MEDICAL X-RAY IMAGE DATASET  </b>
# AASD 4015 - Advance Mathematical Concepts for Deep Learning Group Project 2

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



