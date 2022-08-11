## SentiMerge

### How to get the sentiMerge dictionary:

1. Go to https://github.com/guyemerson/SentiMerge
2. Download the file **sentimerge_nospin.txt** from the **data** folder
3. Remove all lines with **0.0** from the file, e.g., by typing the following command into the command line:

cat sentimerge_nospin.txt | grep -v "\s0.0\s" >sentimerge_nospin_no-null.txt

Make sure that the new file has exactly the name specified above and is in the same folder as the readme file.
