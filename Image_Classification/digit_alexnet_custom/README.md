
## Custom CNN to recognize number

Download dataset MNIST.     
Run `download_dataset.py`
What is the MNIST?  https://en.wikipedia.org/wiki/MNIST_database
<img width="2520" height="1310" alt="image" src="https://github.com/user-attachments/assets/46131058-79fc-426f-97c1-061edaec94fc" />

Build a custom convolutional neural network to recognize which digit it is.
<img width="940" height="398" alt="image" src="https://github.com/user-attachments/assets/092dcfb8-6d51-4e51-9057-ca54e07dd92a" />

The net class is this `net_class.py`:       
<img width="2646" height="1048" alt="image" src="https://github.com/user-attachments/assets/26953047-fbaa-42a4-a372-d8bdcd08064a" />


Run `main_train_save_model.py` to train a modle, name best_model.pth in root folder.
<img width="582" height="800" alt="image" src="https://github.com/user-attachments/assets/db3bddf9-88ec-482e-a6f9-fdea8862c158" />


Test modle.     
Run `main_test_model.py`
<img width="2166" height="148" alt="image" src="https://github.com/user-attachments/assets/7fc3ebb4-2e8f-4efe-ab22-d87a0992611f" />

Recognize a number, such as 3.       
Run `identify_number.py`
<img width="1888" height="1954" alt="image" src="https://github.com/user-attachments/assets/3eb8393e-8042-4727-8ff2-47b40ed03efa" />
