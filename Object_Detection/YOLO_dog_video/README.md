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
<img width="574" height="338" alt="image" src="https://github.com/user-attachments/assets/3b3b90cb-f297-4e4f-ab6a-c1e51ad32f1f" />
Output `Video_data\Animal_filtered.mp4`
<img width="575" height="288" alt="image" src="https://github.com/user-attachments/assets/58cae7fd-42f3-4be2-96d3-dd216bf535cd" />
     
In the MP4 result file, detect the dogs.
<img width="1098" height="456" alt="image" src="https://github.com/user-attachments/assets/cd15ba9b-601d-42a6-8885-c29474c2a0e5" />

