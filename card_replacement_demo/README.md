This is a corner-features based homography estimation / planar object recognition and replacement demo.

![Screenshot](https://github.com/Duffycola/demos/blob/master/card_replacement_demo/screenshot.png)

There is both a python and C++ variant.



Running the python demo:

python card_replacement_demo.py --help
python card_replacement_demo.py sequence/
python card_replacement_demo.py sequence/ --debug

Compiling and running the C++ code:

g++ card_replacement_demo.cpp -std=c++11  -o card_replacement_demo `pkg-config --cflags --libs opencv`
find sequence/*.jpg | ./card_replacement_demo 
