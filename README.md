# VSSS
The source code of **Vine Spread for Superpixel Segmentation** (https://ieeexplore.ieee.org/document/10015675)

![example](https://github.com/zach-pei/VSSS/blob/main/pic/vsss-detail-twigs.png)

# Get Started

First, make a new directory to build the whole project.

And make sure you are in the VSSS project root directory.
```bash
mkdir build
cd build
cmake ..
make
```
Then the executable file will be in the ./bin directory.

Finally, run the command below to segment superpixel.

```bash 
./bin/main --input ./pic/ --num_sp 1000 --output ./out --output_sp --output_label --alpha 0.005 --lambda 20 --beta 30 --tau 7
```

