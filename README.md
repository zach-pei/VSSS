# VSSS
The source code of Vine spread for Superpixel Segmentation


# Get Started

First, make a new dir to build the whole project.

And make sure you are in the VSSS project root dir.
```bash
mkdir build
cd build
cmake ..
make
```
Then the executable file will be in the ./bin dir.

Finally, run the command bellow to segment superpixel.

```bash 
./bin/main --input ./pic/ --num_sp 1000 --output ./out --output_sp --output_label --alpha 0.005 --lambda 20 --beta 30 --tau 7
```
