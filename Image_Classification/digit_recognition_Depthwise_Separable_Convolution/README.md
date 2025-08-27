
## Custom CNN to recognize number
Use Depthwise Separable Convolution.    
Depthwise Separable Convolution is used to improve the efficiency of CNN and significantly speed up the training process.

Download dataset MNIST.     
Run `download_dataset.py`
What is the MNIST?  https://en.wikipedia.org/wiki/MNIST_database
<img width="2520" height="1310" alt="image" src="https://github.com/user-attachments/assets/46131058-79fc-426f-97c1-061edaec94fc" />

Build a custom convolutional neural network to recognize which digit it is.
<img width="940" height="398" alt="image" src="https://github.com/user-attachments/assets/092dcfb8-6d51-4e51-9057-ca54e07dd92a" />


Run `main_train_save_model.py` to train a modle, name best_model.pth in root folder.
<img width="2164" height="1406" alt="image" src="https://github.com/user-attachments/assets/989e2c92-4a1b-4db7-82a4-ccbe23c63681" />
The best_model.pth is small.
<img width="1542" height="502" alt="image" src="https://github.com/user-attachments/assets/2b217776-a9fd-4f66-8c32-53e439d042f2" />


Recognize a number, such as 2.       
Run `identify_number.py`
<img width="2248" height="1742" alt="image" src="https://github.com/user-attachments/assets/0888d53f-e55a-4b03-93b9-390f71633dc8" />

