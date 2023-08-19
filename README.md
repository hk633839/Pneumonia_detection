# Pneumonia_detection

This is a deep learning model implementation of detecting the presence of pneumonia in patients.

The training images are chest x-ray images.

Variables used :

Image size : 299 * 299 

Batch size : 128

Initial Bias : log(Positive samples / Negative samples)

## Model architecture : ##

 Layer (type)               | Output Shape           |   Param 
 ---------------------------|------------------------|---------
 Conv2D         | (299, 299, 16)   |   448                 
 Conv2D         | (299, 299, 16)   |   2320            
 MaxPooling2D | (149, 149, 16)   |   0                                                                     
 Convolution Block(32) | (74, 74, 32)     |   2160      
 Convolution Block(64) | (37, 37, 64)     |   7392      
 Convolution Block(128) | (18, 18, 128)    |   27072     
 Dropout(0.2)       | (18, 18, 128)     |  0         
 Convolution Block(256) | (9, 9, 256)       |  103296    
 Dropout(0.2)       | (9, 9, 256)       |  0         
 Convolution Block(128) | (4, 4, 128)       |  53376     
 Dropout(0.2)       | (4, 4, 128)       |  0         
 Flatten        | (2048)            |  0         
 Dense Block(512, 0.7) | (512)             |  1051136   
 Dense Block(128, 0.5) | (128)             |  66176     
 Dense Block(64, 0.3) | (64)              |  8512      
 Dense           | (1)               |  65        

#### Conv2D layer arguments ####
Conv2D(filters = 16, kernel_size = 3, activation = 'relu', padding = 'same')

#### Convolution Block Architecture and arguments ####
↣ SeparableConv2D(filters, 3, activation='relu', padding='same')

↣ SeparableConv2D(filters, 3, activation='relu', padding='same')

↣ BatchNormalization()

↣ MaxPool2D()

#### Dense Block Architecture and arguments ####
↣ Dense(units, activation='relu')

↣ BatchNormalization()

↣ Dropout(dropout_rate)

                                                                 
Total params: 1321953 (5.04 MB)  
Trainable params: 1319329 (5.03 MB)  
Non-trainable params: 2624 (10.25 KB)

### Noraml Chest X-ray: ###

![IM-0117-0001](https://github.com/hk633839/Pneumonia_detection/assets/97289725/7a598702-8305-4d27-b2cb-b779ef2063c6)

### Pneumonia Chest X-ray: ###

![person2_bacteria_4](https://github.com/hk633839/Pneumonia_detection/assets/97289725/821ba845-39ed-4680-91b5-b2ad8dbbd653)

The dataset can be found here : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Model Performance ##

![output](https://github.com/hk633839/Pneumonia_detection/assets/97289725/4343c1cc-c3a4-4b14-af60-7f93a8b0085f)

