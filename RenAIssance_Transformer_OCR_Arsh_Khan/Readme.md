![](https://github.com/humanai-foundation/RenAIssance/blob/main/RenAIssance_Transformer_OCR_Arsh_Khan/images/humanai.png)

# <ins>Table of Contents

* [About the Project](#about-the-project)
* [Printing Irregularities](#printing-irregularities)
* [Theory and Approach](#theory-and-approach)
  * [Data Preprocessing](#data-preprocessing)
  * [Training the model](#training)
  * [Inference](#inference)
* [Results](#results)
* [Useful Links](#links)
* [Acknowledgements](#acknowledgements)
* [License](#license)

---

# <ins>About the Project

**This project aims to develop a hybrid end-to-end Transformer model capable of accurately recognizing text from non-standard Spanish printed sources from the 16th and 17th centuries. It was developed as part of the Google Summer of Code (GSoC) initiative.**

---
# <ins> Printing Irregularities

Historical texts from the 16th and 17th centuries present a variety of printing irregularities that significantly challenge Optical Character Recognition (OCR) systems. These irregularities arise from the printing limitations and linguistic conventions of the period. Our model specifically addresses these challenges to improve text recognition accuracy. Below are some common irregularities and how they are managed:

1. **Interchangeable Characters**:
   - **`u` and `v`**: In historical texts, these letters are often used interchangeably. Typically, 'u' appears at the beginning of a word, while 'v' is used within a word. The model is trained to adapt to this usage pattern by recognizing context-specific rules.
   - **`f` and `s`**: Both letters can be used interchangeably. The model assumes 's' at the beginning or end of a word and 'f' within a word, learning these patterns from annotated training data.

2. **Tildes and Accents**:
   - Tildes serve as horizontal caps. When a `q` is capped, `ue` usually follows, while a capped vowel is typically followed by `n`. A capped `n` is always interpreted as `ñ`.

3. **Spelling Conventions**:
   - **`ç` to `z`**: The old spelling `ç` is interpreted as the modern `z`. Historical spelling conventions are mapped to contemporary equivalents to enhance text accuracy.

4. **Hyphenation**:
   - Line-end hyphens are often missing, resulting in split words. The model retains these splits initially and uses dictionary checks to ensure accurate reconstruction, correcting split words during post-processing.

### Handling Irregularities

To effectively manage these printing irregularities, the project employs a combination of strategies:

- **Customized Training Data**: The training dataset includes annotated examples of these irregularities to help the model learn context-specific corrections.

- **Augmentation Techniques**: Data augmentation strategies introduce variability similar to historical irregularities, enhancing the model's robustness.

- **Post-Processing Rules**: After initial recognition, a post-processing phase applies specific rules to correct identified irregularities. This includes using historical dictionaries and linguistic rules to refine the output.

- **Contextual Awareness**: The model leverages contextual information from surrounding text to make informed decisions about character interpretation and word segmentation.

These strategies enable the model to overcome the complexities of historical printing irregularities, improving the overall accuracy and reliability of OCR results for early Spanish printed sources. This work not only advances the preservation of historical documents but also enhances their accessibility for modern research.

---
# <ins>Theory And Approach

## Data Preprocessing

Data preprocessing is a critical step in historical document text recognition, as the quality of preprocessed data significantly impacts model performance. The preprocessing pipeline for this project involves several stages to ensure optimal data quality and usability:

1. **PDF to Image Conversion**: Historical documents in PDF format are converted into high-resolution images and split into individual pages.

2. **Image Preprocessing**:
   - **Deskewing**: Corrects the orientation of the text to ensure it is aligned horizontally.
   - **Denoising**: Removes noise from images to enhance clarity.
   - **Binarization**: Converts images to a binary format (black and white) using techniques like Otsu’s algorithm.
   - **Increasing DPI**: Improves image resolution to enhance the accuracy of text recognition.
   - **Contrast Enhancement**: Adjusts image contrast to improve text visibility.
   - **Normalization and Resizing**: Standardizes image sizes and pixel values to ensure consistency across the dataset.

3. **Layout Analysis and Line Segmentation**:
   - **UNet Segmentation**: Used for layout analysis to identify main text regions.
   - **Astar Path Planning Algorithm**: Applied for line segmentation to identify individual text lines. This involves calculating horizontal projection profiles and applying path planning algorithms to accurately segment text lines.

4. **Data Augmentation**: 
   - Various augmentation techniques are applied, including **Gaussian Noise, Optical Distortion, CLAHE, Affine Transformations, Perspective Transformations, and Elastic Transformations.**
   - Augmentation increases the diversity of the training dataset, helping the model generalize better by exposing it to various versions of the same images.

## Training

The training process involves several key components to ensure the model effectively learns from the data and achieves high accuracy in OCR tasks:

1. **Model Architecture**: 
   - The model is based on the TrOCR (Transformer Optical Character Recognition) and Vision Transformer (ViT) architectures.
   - Transformers are used for their ability to handle sequences and capture long-range dependencies in data through self-attention mechanisms.

2. **Fine-Tuning Strategies**:
   - **Gradient Accumulation**: Accumulates gradients over multiple mini-batches before updating model weights. This technique allows training with larger effective batch sizes without exceeding memory constraints.
   - **AdamW Optimizer**: A variant of the Adam optimizer that decouples weight decay from the optimization step, helping to improve generalization and prevent overfitting.
   - **Learning Rate Schedulers**:
     - **Cosine Scheduler**: Linearly increases the learning rate during a warm-up phase and then applies a cosine decay towards 0 for the remaining training steps. This smooth reduction helps improve convergence without abrupt changes.
     - **Cosine Annealing**: Periodically decays and resets the learning rate in cycles, allowing the model to avoid local minima and promoting better exploration of the loss landscape, which can improve generalization and performance.

3. **Regularization Techniques**:
   - **Dropout**: Randomly deactivates neurons during training to prevent overfitting.
   - **Weight Decay**: Penalizes large weights to encourage simpler models and improve generalization.
   - **Label Smoothing**: Softens target labels to prevent the model from becoming overconfident in its predictions.
   - **Temperature Scaling**: It involves adjusting the logits of the output layer by a scalar temperature parameter to produce more calibrated probability estimates.

4. **Warmup Ratio**:
   - Gradually increases the learning rate from a small initial value to the target learning rate over a few iterations at the start of training. This stabilizes the training process and improves convergence.

5. **Early Stopping**:
   - Monitors the model’s performance on a validation set and halts training when performance ceases to improve, preventing overfitting and saving computational resources.

6. **Evaluation Metrics**:
   - The model’s performance is continuously monitored using Character Error Rate (CER), Word Error Rate (WER), and BLEU Score during training to guide adjustments to hyperparameters and strategies.

7. **Loss Function**:
Various loss functions have been experimented and implemented to help the model fine-tune across different data distributions, to enhance both its accuracy and generalizability.

   - **Beam Search Loss**: Integrates beam search decoding into the training process, focusing on generating coherent and contextually relevant sequences. Includes refinements like length normalization and normalized log-likelihood objectives to ensure fair evaluation of sequence lengths.
   - **Focal Loss**: Focal Loss is a modification of cross-entropy loss designed to address class imbalance by focusing more on hard-to-classify examples. It introduces a scaling factor, where pt is the predicted probability of the true class and γ is a tunable parameter (called the focusing parameter). This factor down-weights the contribution of easily classified examples, allowing the model to focus more on difficult or misclassified instances.

By following this detailed preprocessing and training strategy, the Transformer OCR model is designed to achieve high accuracy and robust performance in recognizing complex historical Spanish texts.

## Model Calibration
Several Heuristics such as **label smoothing, beam search decoding, length normalization, and trigram blocking** have been implemented. However, these methods fall short as they are based on indirect supervision, influencing predictions without explicitly optimising the model for accurate probability distributions, leaving the problem of uncalibrated sequence likelihood unresolved.

SLiC represents a natural extension of the current pretraining and fine-tuning paradigm, offering a more calibrated and robust approach to sequence generation, by calibrating sequence likelihoods directly in the model’s latent space.

The SLiC objective function consists of two components: **Calibration Loss** and **Regularization Loss**.

- **Calibration Loss**: This aims to align the model's predictions with the target sequence similarity. Two loss types are used for its implementation:
  - **Rank Loss**: Ensures that positive candidates are ranked higher than negative ones.
  - **Margin Loss**: Increases the probability gap between positive and negative candidates.

- **Regularization Loss**: This prevents the model from diverging too far from its original objective. KL Divergence is used for regularization to maintain consistency in the model's predictions.

Additionally, two other loss types are considered:
  - **List-wise Rank Loss**: Optimizes the overall ranking of a list of candidates.
  - **Expected Reward Loss**: Maximizes the expected similarity across all candidate sequences.


## Inference

Inference is the final stage where the trained model is used to recognize text from unseen historical document images. The goal is to accurately transcribe these images into machine-readable text. Here are the key steps and considerations for the inference process:

1. **Preprocessing for Inference**:
   - **Image Preprocessing**: Similar to the training phase, input images are preprocessed by deskewing, denoising, binarization, contrast enhancement, normalization, and resizing to ensure consistency and quality.
   - **Line Segmentation**: Uses UNet and A* Path Planning algorithms to segment text lines for individual processing. This segmentation is crucial for accurately recognizing text from complex historical layouts.

2. **Model Inference**:
   - The trained TrOCR model processes each segmented line image to generate a sequence of characters or words. 
   - The model employs a beam search decoding strategy to generate the most probable transcription. This involves maintaining multiple hypotheses (beams) during decoding and selecting the one with the highest overall probability.

3. **Post-Processing**:
   - **Text Correction**: Post-processing steps are applied to refine the output text. This includes correcting common OCR errors, such as those involving historical orthographic conventions and interchangeable characters.
   - **Dictionary and Grammar Checks**: Utilizes dictionaries and grammar rules specific to the historical context to correct spelling and grammatical errors.

4. **Performance Metrics**:
   - The quality of the model’s output during inference is evaluated using Character Error Rate (CER), Word Error Rate (WER), and BLEU Score. These metrics provide insight into the accuracy and readability of the transcriptions.
   - Levenshtein Distance, may also be used to quantify the dissimilarity between the predicted and ground truth text strings, providing another measure of transcription accuracy.

5. **Output**:
   - The final output is a transcribed text file, where the historical document's content is digitized and accessible for further analysis or preservation.
   - The transcribed text can be saved in various formats, such as plain text, CSV, or JSON, depending on the user's requirements and downstream applications.

6. **Visualization**:
   - For visual inspection, the model can output images with overlaid transcriptions to verify alignment and accuracy.
   - This helps in identifying areas where the model may need further fine-tuning or adjustment.

The inference pipeline is designed to be efficient and accurate, leveraging the model’s learned capabilities to handle the complexities of historical Spanish documents. By integrating sophisticated preprocessing, decoding, and post-processing techniques, the system aims to provide high-quality OCR results that facilitate research and preservation efforts.

---
## <ins> Results

The TrOCR model demonstrates significant improvements in OCR performance, especially for historical Spanish texts. Current Key performance metrics include:

- **Character Error Rate (CER):** 0.03 (97% accuracy)
- **Word Error Rate (WER):** 0.07 (93% accuracy)

--- 
## <ins>Useful Links

For more comprehensive details about the project and documentation, you can explore the following links:

- **Blog** : [Arsh Khan Transformer OCR Blog](https://medium.com/@khanarsh0124/gsoc-2024-with-humanai-text-recognition-with-transformer-models-de86522cdc17)

- **Google Summer of Code (GSoC) 2024 Project**:   
  [GSoC 2024 Project](https://summerofcode.withgoogle.com/programs/2024/projects/qnIVjbSY)

- **HumanAI Foundation**:[HumanAI Projects](https://humanai.foundation/activities/gsoc2024.html)

Contributions to this project are welcome. If you are interested in contributing, please fork the repository and submit a pull request. Ensure that your code follows the style and conventions used in this repository.

---

## <ins>Acknowledgements

This project was developed as part of the **Google Summer of Code (GSoC)** initiative, with support from the **HumanAI Foundation** and contributions from the open-source community.

---
## License

This project is licensed under the MIT License. See the [MIT Licence](https://github.com/humanai-foundation/RenAIssance/blob/main/RenAIssance_Transformer_OCR_Arsh_Khan/LICENCE)
