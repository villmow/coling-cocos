# COCOS

This is the repo to our COLING 2022 paper [Addressing Leakage in Self-Supervised Contextualized Code Retrieval](https://aclanthology.org/2022.coling-1.84/).

We address contextualized code retrieval, the search for code snippets helpful to fill gaps in a partial input program. 

## Contextualized Code Retrieval

Below is an example of a contextualized code retrieval task. The input program is incomplete and the goal is to find 
code snippets that help to fill the gap.
```java
public boolean extract(File f, String folder) {
   Enumeration entries; ZipFile zipFile;
   try {
       zipFile = new ZipFile(f);
       entries = zipFile.getEntries();
       ___MASK___
       zipFile.close();
   } catch (IOException ioe) {
       this.errMsg = ioe.getMessage();
       Malgn.errorLog("{Zip.unzip} " + ioe.getMessage());
       return false;
   }
   return true;
}
```
This is a snippet that would be helpful.
```java
ArchiveEntry ae = zis.getNextEntry();
while(ae != null) {
   //Resolve new file
   File newFile = new File(outputdir + File.separator + ae.getName());
   //Create parent directories if not exists
   if(!newFile.getParentFile().exists()) 
        newFile.getParentFile().mkdirs();
   if(ae.isDirectory()) { //create if not exists
       if(!newFile.exists()) newFile.mkdir();
   } else { //If file, write file
       FileOutputStream fos = new FileOutputStream(newFile);
       int len;
       while((len = zis.read(buffer)) > 0) {
           fos.write(buffer, 0, len);
       }
       fos.close();
   }
   //Proceed to the next entry in the zip file
   ae = zis.getNextEntry();
}
```

## Installation

It requires `pytorch` to be installed, all other dependencies should be installed automatically.

Change to the root of this repo and install the package with pip (either editable or not):
```bash
pip install --editable .
```

Here is how I setup a new conda environment for this project:
```bash
conda create --name cenv-cocos python=3.11 
conda activate cenv-cocos

# get pytorch installation command from https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install --editable .
```

#### Troubleshooting
If you receive any tree-sitter related errors, you might need to install the tree-sitter grammars manually.
Run the following command to download the tree-sitter grammars:
```bash
rm -r cocos/tokenizecode/libs
python -c "import cocos; t = cocos.tokenizecode.CodeParser()"
```

## Experiments

There are detailed instructions for each experiment in the corresponding experiment directory. 

 - [02 COCOS evaluation](experiments/02-cocos_evaluation/readme.md)
 - [03 Defect detection](experiments/03-defect_detection_(devign)/readme.md)
 - [04 Clone detection](experiments/04-clone_detection_(poj104)/readme.md)

## Cite
If you find this work useful, please cite our paper:

```bibtex
@inproceedings{villmow-2022-cocos,
    title = "Addressing Leakage in Self-Supervised Contextualized Code Retrieval",
    author = "Villmow, Johannes  and
      Campos, Viola  and
      Ulges, Adrian  and
      Schwanecke, Ulrich",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.84",
    pages = "1006--1013",
    abstract = "We address contextualized code retrieval, the search for code snippets helpful to fill gaps in a partial input program. Our approach facilitates a large-scale self-supervised contrastive training by splitting source code randomly into contexts and targets. To combat leakage between the two, we suggest a novel approach based on mutual identifier masking, dedentation, and the selection of syntax-aligned targets. Our second contribution is a new dataset for direct evaluation of contextualized code retrieval, based on a dataset of manually aligned subpassages of code clones. Our experiments demonstrate that the proposed approach improves retrieval substantially, and yields new state-of-the-art results for code clone and defect detection.",
}
```