First install the components: pip install transformers ultralytics
Then install hugging-face
<img width="940" height="164" alt="image" src="https://github.com/user-attachments/assets/227d3be9-3155-461b-ac73-c05b61194ca0" />

The YOLOv8 image recognition model is used for fast and accurate target detection, and the CLIP model (Contrastive Language-Image Pre-training) is used to understand the semantic relationship between images and text.
When you run CLIP for the first time, the model file will be automatically downloaded from Huggingface. When deploying on the cloud, you must be careful about the read and write permissions.
<img width="940" height="234" alt="image" src="https://github.com/user-attachments/assets/c1085665-83ca-4921-8436-47abc4998868" />

Run the model, find top n similar cars. 
<img width="2142" height="1862" alt="image" src="https://github.com/user-attachments/assets/be545590-d334-4125-b41e-b5c841ccc2bd" />


Results: The search for yellow vehicles successfully detected the three yellow vehicles with the highest matching scores.
<img width="940" height="653" alt="image" src="https://github.com/user-attachments/assets/1c66e060-e8f3-4077-91df-ff578fe5bf55" />

Find green car in the other test image.   
<img width="1834" height="1012" alt="e60016a3bd1224b34d2aedd1bea3f450" src="https://github.com/user-attachments/assets/7ef34fd6-6e2c-4480-9b3b-eeed81155690" />
Find black cars in the other test image.   
<img width="1816" height="1052" alt="582b0de0094f35740e0c6f5a6ca2aab3" src="https://github.com/user-attachments/assets/02be36d0-b6ad-44c4-8807-1c32570f9cec" />
Find blue cars in the other test image.  
<img width="1850" height="1062" alt="b3aef8065e2f949e16882ac525a0fe58" src="https://github.com/user-attachments/assets/e8d7366d-0bcb-42fc-a52e-aecf03868a62" />

