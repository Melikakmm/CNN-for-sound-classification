# SoundFusion: Music Recognition by Information Fusion of Different Audio Representations


In this project, we delve into the exciting world of music classification - a challenge that requires the identification of a musical piece's genre or style. We approach this task in a dual way, by utilizing the information provided by the raw signal itself and also by taking advantage of the spectrogram representation of the tracks, leveraging the powerful capabilities of Convolutional Neural Networks (CNNs) as image classifiers.We present a comprehensive music classification system using various CNN architectures implemented in PyTorch, including ResNet34 and ResNet18 (both with pretrained and random weight initialization), and our own custom ResNet34 architecture with added regularization techniques for improved performance. Additionally, we propose our own Simple CNN for comparison. Our approach utilizes both raw audio signals and spectrogram representations as inputs to the CNNs, with the goal of accurately classifying music into eight predefined genres.
The focus of our paper is twofold: (1) the creation of our own custom preprocessing class to prepare the data before feeding it to the model, (2) a comparative analysis of the state-of-the-art models on the FMA dataset, and, (3) the SoundFusion architecture that we propose and which can learn from different audio representations. We provide in the report a summary of the results to showcase the effectiveness and limitations of using CNNs in music classification tasks.


# Results 

To visualize the results, you can check our report that you can find in this repository.
Several results are available here:
https://docs.google.com/spreadsheets/d/1sCCcPoR8EBBya6jRyCnfTytkSTWD2QRgWzJlIAxwu5s/edit#gid=0



![Test Image 8](k.jpeg)




