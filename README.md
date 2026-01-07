This is the code for our CUDA optimization project, where we look into how we can further optimize Roger Allen's CUDA implementation of Ray Tracing in One Weekend, a raytracing book/engine by Peter Shirley, Trevor David Black and Steve Hollasch.

On unix systems (like Colab), the code can be built and ran using the MAKEFILE that came with the project.

# To build and output image to a file, use:

make out.ppm

This results in a PPM image file, which is old an inefficient. Since the program prints both the image pixels and the text to standard output, it can sometimes become jumbled. Therefore we chose to output the image with the following python script.

---

from PIL import Image

try:
img = Image.open('out.ppm')
display(img)
except FileNotFoundError:
print("image not found")

---

Note: SM architecture needs to be set in the MAKEFILE, and can be found by searching for your GPU on Google or with NVIDIA SMI

# To profile with nvprof, use

make profile_basic

# To profile with NCU, use

ncu --log-file report.txt ./cudart > (some null directory)
cat report.txt

The null directory above is just somewhere to discard the output image, since we do not actually need it for profiling
