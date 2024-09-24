![thumbnail](https://github.com/user-attachments/assets/b0aa865c-416c-4a3a-92be-56a1a77c8f4e)
# RenAIssance
The analysis of historical documents is a critical yet costly method in the Humanities. To reduce these costs, AI technology, specifically OCR (Optical Character Recognition), has started to be utilized. However, for many years, there was a lack of accurate OCR tools for Spanish documents from the Renaissance period, despite their academic importance. To address this issue, the HumanAI Foundation launched the **RenAIssance** project, where contributors implement accurate OCR models using various approaches.

# Dataset 
![letters](https://github.com/user-attachments/assets/c10584db-8f68-4897-a6c4-c70411ed9515)

The dataset used to train these models consists of images of printed documents from the target era, collected from diverse sources. A portion of the data has been manually labeled by RenAIssance mentors, who are experts in Spanish historical documents. The following printing irregularities in the data present challenges for creating high-accuracy OCR models:

- **Interchangeable Characters:** Characters such as 'u' and 'v', and 'f' and 's' were often used interchangeably.
- **Tildes and Diacritical Marks:** Used to save space or due to the reuse of type molds.
- **Old Spellings and Modern Interpretations:** Variations in character usage between historical and modern Spanish.
- **Line-End Hyphens:** Words split across lines were not always hyphenated.

Additionally, the deterioration and unique layouts of historical documents further complicate OCR tasks, making content extraction from images difficult.

# Method  
To address these challenges, contributors have introduced various state-of-the-art (SOTA) methods. These can be broadly classified into the following three approaches:

1. **CRNN Approach**  
2. **Vision Transformer Approaches**  
3. **Self-Supervised Learning Approach**  

All models, regardless of the approach used, achieve over 90% accuracy. For more detailed information on each approach, please refer to the contributors' repositories.
