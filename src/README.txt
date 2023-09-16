No argument is needed. Just type 'python ./script.py' to run the script.

Path to input image is hard-coded into the script. Please make sure that 'campus.tiff' is in the same directory as the script, or change the path on line 12 as you may see fit. Output is hard-coded to be 'campus.png' and 'campus.jpeg' under the same directory. Please also change this on line 203-204 as you see fit. For your convenience, I have included 'campus.tiff' in this directory.

Codes used for identifying the Bayer pattern (line 32-55) and performing manual white balancing (line 88-104) are commented out because they interfere with the normal execution of the whole pipeline. Please uncomment if you want to test those parts.

On line 107, you can choose from different white balancing methods by assigning to 'image' one of 'image_whiteWorld', 'image_grayWorld', 'image_preset', and 'image_manual' (if this part of code is uncommented). Currently it is set to 'image_preset', which is one that I like the best.