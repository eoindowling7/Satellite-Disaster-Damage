# Satellite-Based Disaster Damage Assessment 

An end-to-end deep learning and geospatial analysis project for locating buildings in satellite imagery, predicting post-disaster damage severity, and using those predictions to assess population and critical-infrastructure exposure.

![Building-Level Disaster Damage Assessment](figures/satellite_damage_prediction_overlay_filled.png)

## Project Overiview

I built this project to explore how satellite imagery and deep learning could be combined to assess the impact of natural disasters at a large scale. Rather than treating it as a single image-classification problem, I wanted to build a pipeline that starts with raw satellite imagery and ends with information that could actually be useful when assessing an impacted area.

The pipeline first identifies building footprints from satellite imagery using semantic segmentation. Pre and post-disaster imagery is then used together to classify each building based on the severity of damage caused. Once the predictions are finished, the system then uses the image meta data to source coordinates and connect all predictions to their respective geographic locations. Finally, using public records regarding population and infrastructure, mapping is done to represent the damage relative to its human impact.

The overall pipeline is:

**Satellite imagery → Building segmentation → Damage classification → Spatial damage mapping → Population & infrastructure exposure**

A major focus of the project was generalisation. To prioritise realistic performance on completely unseen disasters, I made sure to group all images with their respective events during the dataset split so testing data would never resemble training data and lead to falsely inflated results.

## Dataset and Geographic Coverage

The project uses the [xBD/xview2](https://xview2.org/dataset) dataset, the competition dataset provided paired pre and post disaster satellite imagery along with building footprints and damage class labels for each building. In total the dataset provides approximately 425,000 labelled buildings across over 7,000 image pairs, covering earthquakes, flooding, wildfires, tsunamis, volcanic eruptions and wind-related disasters. I chose the dataset for its great diversity, with the multiple classes of disasters, spread across multiple continents, it provides the truly balanced environment which would be required to deploy a model for real world application.

![Global Distribution of xBD Disaster Events](figures/xbd_disaster_world_map.png)

As mentioned before, I separated the data by disaster event rather than randomly splitting images. This allowed the system to be evaluated based on truly unseen disasters as would happen in real life. However this did come with a downside for the less common disasters, such as earthquakes and tsunamis which only appeared once and twice in the dataset respectively, as they were then forced to be absent from one of the split groups, with the tsunami event being in the training and testing data but not the validation.

## Building Footprint Segmentation

The first stage in the pipeline was to identify the buildings themselves. So in order to assess how badly a building had been damaged, I needed a model capable of separating building footprints from the rest of the image.

I chose to compare three semantic- segmentation architectures;
- U-Net
- DeeplabV3+
- U-Net++
I deliberately chose related architectures which used different approaches to feature extraction and reconstruction, as it would allow me to compare both their performance and how information moves through their encoder-decoder structures.

### Architecture Comparison

Despite all three models having the same output goal they reconstruct the spatial information in completely different ways.

- **U-Net** is the most intuitive, essentially compressing an image to later rebuild it. The encoder gradually reduces the detail of the image while learning the features from the most fine such as the edges and then eventually identifying the entire structures and shapes. The decoder then expands the quality of this compressed representation while decider which pixels belong to target features (our buildings in this case). The features identified in each layer of encoding are passed directly to their corresponding decoder, allowing all features to be identified as the decoder expands back to original quality.

- **U_Nett++** follows the same basic idea for its encoding and decoding, but instead of passing the corresponding information directly from each encoder to decoder, it uses a series of intermediate connections to first refine the information. With the features being gradually combines with one another at several levels before even reaching the decoder. This extra refinement often helps with the small densely packed buildings which may require combined context to accurately identify.

- **DeepLabV3+** takes a different approach. Rather than relying as strictly on this encoder-decoder structure, it looks at the image on multiple spatial scales, essentially examining each feature on the map through multiple fields of view simultaneously. Combining the outputs from multiple scales of view provides insights about the contextual features which may indicate information on you target (for example when searching for buildings it may identify the road running beside them as a pattern, allowing it to find buildings along roads more easily)

![Segmentation Architecture Comparison](figures/segmentation_architecture_comparison.png)

I was particularly interested in the decoder side of the models, as this had to reconstruct the scenes accurately enough to later identify changes in the buildings and recover individual boundaries.

### From Features to Building Masks

To get a better idea of what was happening inside the segmentation models, I also visualised intermediate feature representations rather than treating each network entirely as a black box.

The example below follows the U-Net++ decoder as it progressively reconstructs a building mask. Early decoder features still respond broadly to the structure of the scene, while later stages become increasingly concentrated around individual buildings. The final output converts these reconstructed features into a pixel-level building probability map.

![U-Net++ Decoder Progression](figures/unetplusplus_decoder_progression.png)

This shows how a model is not simply identifying a building once at the end of the network, but rather reconstructing the building mask back up from largest features to most detailed.

### Segmentation Results

The metric of IoU (intersection over union), which is a measurement of how well the model masks the target, was our prioritised measure of performance. After training each model for a baseline of 20 epochs, the U-Net++ showed the most promise, leading me to give it a full 50 epochs of training. The U-Net++ then achieved the highest score of 0.650, comparing to 0.628 and 0.589 for the U-Net and DeepLabV3++ respectively.

![Building Segmentation Performance](figures/validation_iou_model_comparison.png)

This metric supported the use of the U-Net++ model for the next stage of the project. However I wanted to look beyond a single aggregate metric, so I compared the models on the performance within a difficult scene, containing dense clustering of buildings.

![Dense Scene Segmentation Comparison](figures/dense_scene_error_comparison.png)

The models generally recover the larger buildings well, with smaller buildings and exact boundaries proving much more difficult.

## Building Damage Classification

Now that I had a model for successfully identifying buildings, I had to determine how severely each building was damaged. This stage is much harder than simply identifying the buildings as the differences between damage classes can be quite subtle from above, especially when trying to separate minor and major damage.

I figured the best way to do this would be to simultaneously look at both pre and post disaster image crops. This helps identify any subtle changes by referencing the pre disaster appearance. Each building was then classified into one of four categories, being no damage, minor damage, major damage and destroyed.

### Siamese Architecture

I chose to use a Siamese ResNet architecture. The main idea is to pass both pre and post disaster images through the same type of feature extractor, producing compact representations of each image, these are then combined before the final classifier predicts the damage class. Essentially the model is trained to answer "How much did this footprint change?", with the answer being checked against its learned pattern to see what damage class it aligns with.

I started with the ResNet18 as a lightweight baseline before testing the deeper Resnet34. I also experimented with giving the ResNet34 more surrounding context and focal loss oversampling to see whether its interpretations of the less common damage classes could be improved.

### Model Experiments

I compared the models using macro F1 rather than relying mainly on accuracy. The dataset contains far more no-damage buildings than some of the damaged classes, so a model could achieve a fairly strong accuracy while still performing badly on the classes I was most interested in detecting. Macro F1 gives each of the four damage classes equal importance when calculating the overall score.

![Model 2 Performance Data](figures/model_performance_detailed.png)

The best result came from the base Siamese ResNet34 model, which achieved a validation macro F1 of approximately 0.536. Interestingly, the additional experiments did not produce a clear improvement, the larger-context ResNet34 reached around 0.511, while the focal loss/oversampling model reached around 0.509. ResNet18 also remained very competitive at approximately 0.535.

I found this quite interesting as adding more model complexity or more aggressive class-balancing techniques did not actually yield better results on the unseen data. The small difference between the ResNet18 and ResNet34 also suggested that model capacity alone was not the main limitation of the classification task.

### Training Behaviour

Looking at the ResNet34 training history also revealed a fairly clear gap between training and validation performance. Training macro F1 continued to improve while validation performance began to level off, suggesting that the model was increasingly fitting patterns specific to the training disasters rather than improving its ability to generalise to unseen events.

![ResNet34 Training Validation Curve](figures/resnet34_training_validation_curve_professional.png)

Because of this overfitting, I used the checkpoint with the strongest validation performance rather than simply taking the model from the final training epoch. This discrepancy between training and validation shows how difficult it is to teach a model to perform on completely unseen events.
