# computer-vision-project2

Austin Pray

## Output

Output is saved to out.txt

## Programs

### mnist1.py

Compared to the sample program I made the following enhancements:

- formatted to comply with pep8 autoformatting
- use an AdamOptimizer with a learn rate of 5.5e-3

The use of AdamOptimizer brings the results from 90% accuracy to 96% accuracy on most runs

### mnist2.py

Compared to mnist1.py I made the following enhancements

- add a second convolutional layer that maps 32 feature maps to 64
- add a pooling layer between the first convolutional layer and second
- tune AdamOptimizer to 1e-3

### mnist3.py

Made the following enhancements over mnist2.py

- adds a third convolutional layer
- adds a second pooling layer before the convolutional layer
- tune AdamOptimizer learn rate to 1.01e-3

### mnist4.py

- add a fourth convolutional layer with weights 3, 3, 64, 64
- move the second pooling layer to be after the 4th convolutional layer
- tune AdamOptimizer learn rate to be 1.01e-3

### mnist5.py

- add a fifth convolutional layer after the second pooling layer
- tune AdamOptimizer learn rate to be 1.01e-3
