First install the components: pip install transformers ultralytics
Then install hugging-face
<img width="940" height="164" alt="image" src="https://github.com/user-attachments/assets/227d3be9-3155-461b-ac73-c05b61194ca0" />

The YOLOv8 image recognition model is used for fast and accurate target detection, and the CLIP model (Contrastive Language-Image Pre-training) is used to understand the semantic relationship between images and text.
When you run CLIP for the first time, the model file will be automatically downloaded from Huggingface. When deploying on the cloud, you must be careful about the read and write permissions.
<img width="940" height="234" alt="image" src="https://github.com/user-attachments/assets/c1085665-83ca-4921-8436-47abc4998868" />

