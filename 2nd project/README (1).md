# GenAI-TFM-Project-2024

_Project of Research Agents for IRB documents_

## Previous requirements 📋

 1. Clone the repository

```
git clone git@github.com:mmctech/GenAI-TFM-Project-2024.git
```

 2. Create a venv & install the necessary libraries (requirements_temp is currently necessary)

```
python -m venv DocReaderLenAI

.\DocReaderLenAI\Scripts\activate

pip install -r requirements.txt

pip install -r requirements_temp.txt
```

 3. Add your own OpenAI API key to a new file called ```config/KEYS.py```

```
openai_key = "XXXXXXXXXXXXX"
```

 4. If necessary, update ```.gitignore``` with your own sensitive files


## Folder Explanation :file_folder: 

 + **config:** Folder containing the api keys, the questions to be answered, the config parameters for the OpenAI client and the constants used along the rest of the source files. 

 + **data:** Folder containing all the inputs used for the model, as well as the output generated.

 + **src:** Folder containing the core of the model. It has a utils folder with the functions used divided by use, as well as the main code.

 + **vold:** Folder containing the previous versions of the project.

    + **V0:** Tomás's version.

    + **V1:** Tomás's version with a few improvements:

        + Code more divided by use and organization of own folders.

        + Update of deprecated functions.

        + Inclusion of prompting techniques in order to obtain the confidence level in the response, as well as the reference, indicating also the role of the model.


## Implementation :computer: 

If it is desired to run the code in order to test it, by running the ```src/__main__.py``` file it is possible to obtain a _.xlsx_ output with the questions and answers.

## Previous Work 

The main versions are:

 + **V0:** Tomás's version.

 + **V1:** Code division and ordering. Addition of basic prompt engineering and few shots. Inclusion of 2 different models (RetrievalQA and RetrievalQAWithSourcesChain), with an unique prompt and example.

## Current Improvement Work 🔧

The project is now facing the begginingg of the V1.5, as well as the basic approach for the V2. The main ideas are:

 + V1.5: 
 
   + Fix the confidence levels, using ground truth answers and metrics.

   + Improve the citation information using metadata.

 + V2: 
 
   + Improve the citation information more if needed.

   + Change the chunking division:

      + Iterative variable size using the text format to separate the unrelated paragraphs.

      + Manual size decision using specific keywords.

