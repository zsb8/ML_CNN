
## Custom CNN to recognize number

Download dataset MNIST.     
Run `download_dataset.py`
What is the MNIST?  https://en.wikipedia.org/wiki/MNIST_database
<img width="2520" height="1310" alt="image" src="https://github.com/user-attachments/assets/46131058-79fc-426f-97c1-061edaec94fc" />

Build a custom convolutional neural network to recognize which digit it is.
<img width="940" height="398" alt="image" src="https://github.com/user-attachments/assets/092dcfb8-6d51-4e51-9057-ca54e07dd92a" />

We use PyTorh Alexnet lib. Code is easier. 
<img width="2114" height="926" alt="image" src="https://github.com/user-attachments/assets/f8dce60e-1cf3-4269-974b-2f4aa76aa2e6" />

Run `main_train_save_model.py` to train a modle, name best_model.pth in root folder.
Will run more time. 

The modle file `best_model.pth` is very big, about  144M , more the the file Github limit. 



Recognize a number, such as 3.       
Run `identify_number.py`

