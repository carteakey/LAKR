http://localhost:7474/browser/
http://nlpprogress.com/english/coreference_resolution.html
https://amazon-reviews-2023.github.io/
https://aws.amazon.com/neptune/knowledge-graphs-on-aws/
https://blog.langchain.dev/constructing-knowledge-graphs-from-text-using-openai-functions/
https://cseweb.ucsd.edu/~jmcauley/datasets.html
https://gist.github.com/Tostino/44bbc6a6321df5df23ba5b400a01e37d
https://github.com/HotBento/KG4RecEval
https://github.com/huzaifakhan04/exploratory-data-analysis-on-amazon-review-data-using-mongodb-and-pyspark
https://github.com/hyp1231/AmazonReviews2023
https://github.com/RiyaVachhani/SentimentAnalysis-AmazonReviews/blob/main/Sentiment%20%20Analysis%20on%20Amazon%20Reviews.ipynb
https://github.com/RUCAIBox/LC-Rec
https://github.com/RUCAIBox/LLMRank
https://github.com/RUCAIBox/RecSysDatasets
https://github.com/WLiK/LLM4Rec-Awesome-Papers
https://github.com/yuh-yang/KGCL-SIGIR22
https://graphacademy.neo4j.com/courses/llm-knowledge-graph-construction/
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
https://medium.com/@arhamrumi/amazon-reviews-eda-662c485ec00c
https://paperswithcode.com/
https://python.langchain.com/v0.1/docs/use_cases/graph/semantic/
https://towardsdatascience.com/leverage-keybert-hdbscan-and-zephyr-7b-beta-to-build-a-knowledge-graph-33d7534ee01b
https://ucsd.edu/
https://www.amazon.science/blog/building-commonsense-knowledge-graphs-to-aid-product-recommendation
https://www.kaggle.com/code/arhamrumi/amazon-reviews-eda
https://www.kaggle.com/code/mohamedbakrey/eda-for-amazon-product-review-sentiment-analysis
https://www.linkedin.com/pulse/amazon-review-sentiment-analysis-report-muhammad-shayan-umar-ukfkf/
https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
https://www.overleaf.com/project/65de2f0b740333a9295fe046
https://www.reddit.com/r/LocalLLaMA/comments/186qq92/using_mistral_openorca_to_create_a_knowledge/
https://www.torontomu.ca/content/dam/early-childhood-studies/pdfs/student-resources/Graduate/mrp-guidelines.pdf
https://www.torontomu.ca/graduate/datascience/faculty/#!accordion-1499783917273-faculty-of-engineering-and-architectural-science
https://www.torontomu.ca/graduate/datascience/students/MRPguidelinesforms/
https://www.torontomu.ca/graduate/datascience/students/MRPs/
https://yupenghou.com/
https://github.com/aws/graph-notebook
https://amazon-reviews-2023.github.io
https://github.com/fusion-jena/automatic-KG-creation-with-LLM


conda create -n llmag python=3.10
conda activate llmag
pip install neo4j

# Enable the visualization widget
jupyter nbextension enable  --py --sys-prefix graph_notebook.widgets

# copy static html resources
python -m graph_notebook.static_resources.install
python -m graph_notetbook.nbextensions.install

# copy premade starter notebooks
python -m graph_notebook.notebooks.install --destination ~/notebook/destination/dir

# create nbconfig file and directory tree, if they do not already exist
mkdir ~/.jupyter/nbconfig
touch ~/.jupyter/nbconfig/notebook.json

# start jupyter notebook
python -m graph_notebook.start_notebook --notebooks-dir ~/notebook/destination/dir



docker run \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/neo4j/data:/data \
    neo4j
    --env='NEO4JLABS_PLUGINS=["apoc"]'


    conda config --set channel_priority flexible

    conda create -n rapids-24.06 -c rapidsai -c conda-forge -c nvidia      rapids=24.06 python=3.11 cuda-version=12.2     jupyterlab dash pytorch^C


    CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install pandas numpy scikit-learn scipy implicit

     We consider the sequential recommendation task. Given the historical interaction sequence of one user {i1, . . . , il}, the task is to predict the next item of interest il+1, where l is the length of the interaction sequence for the user. The items in the interaction sequence have been ordered chronologically. Here, each item i is associated with a sentence that represents the metadata of the corresponding item. Note that in practice, the items in the interaction sequences and the items to predict are usually from the same domain defined in the Amazon Reviews datasets (Kang and McAuley, 2018; Hou et al., 2022).



# Create 5 sample records
