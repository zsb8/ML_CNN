First install the components: pip install transformers ultralytics
Then install hugging-face
<img width="940" height="164" alt="image" src="https://github.com/user-attachments/assets/227d3be9-3155-461b-ac73-c05b61194ca0" />

The YOLOv8 image recognition model is used for fast and accurate target detection, and the CLIP model (Contrastive Language-Image Pre-training) is used to understand the semantic relationship between images and text.    
<img width="2264" height="746" alt="image" src="https://github.com/user-attachments/assets/6b87a151-3772-4899-b840-bb9b948adb22" />
Use OpenCV to process video.    
<img width="2706" height="998" alt="image" src="https://github.com/user-attachments/assets/a4b6ca5b-d2af-4cc7-9587-f26b2f38deac" />



When you run CLIP for the first time, the model file will be automatically downloaded from Huggingface. When deploying on the cloud, you must be careful about the read and write permissions.
<img width="940" height="234" alt="image" src="https://github.com/user-attachments/assets/c1085665-83ca-4921-8436-47abc4998868" />

After run `main.py`, 

