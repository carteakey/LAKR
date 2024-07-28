## Tasks

- [x] 5 Sample Records
- [x] CF Baseline
- [x] Export triplets script
- [x] KG QUality Metrics
- [ ] add preprocessing to doc
- [ ] remove personal info
- [ ] move documentation to other repo


## KG Quality Metrics
Average Degree: This measures the average number of relationships per node. A higher average degree indicates a more connected graph.
Graph Density: This measures how close the graph is to being complete. It ranges from 0 (no edges) to 1 (fully connected). A higher density indicates a more interconnected graph.
Average Clustering Coefficient: This measures the degree to which nodes in a graph tend to cluster together. It ranges from 0 to 1, with higher values indicating more clustering.


python -m main_kgat --data_name amazon-reviews-23 --data_dir /home/kchauhan/repos/mds-tmu-mrp/datasets --use_pretrain 0 --n_epoch 1 --test_batch_size=500

Reset the db 
sudo reset.sh 

python -m k_core_filtering -k 5 --input_path /home/kchauhan/repos/mds-tmu-mrp/datasets/raw --output_path /home/kchauhan/repos/mds-tmu-mrp/datasets

python -m last_out_split  --input_path /home/kchauhan/repos/mds-tmu-mrp/datasets/5core/rating_only --output_path /home/kchauhan/repos/mds-tmu-mrp/datasets/last_out_split --seq_path /home/kchauhan/repos/mds-tmu-mrp/datasets/last_out_split

We consider the sequential recommendation task. Given the historical interaction sequence of one user {i1, . . . , il}, the task is to predict the next item of interest il+1, where l is the length of the interaction sequence for the user. The items in the interaction sequence have been ordered chronologically. Here, each item i is associated with a sentence that represents the metadata of the corresponding item. Note that in practice, the items in the interaction sequences and the items to predict are usually from the same domain defined in the Amazon Reviews datasets (Kang and McAuley, 2018; Hou et al., 2022).




docker run \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/neo4j/data:/data \
    neo4j
    --env='NEO4JLABS_PLUGINS=["apoc"]'



```
CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install pandas numpy scikit-learn scipy implicit
```


python -m main_kgat --data_name amazon-reviews-23 --data_dir /home/kchauhan/repos/mds-tmu-mrp/datasets --use_pretrain 0  --test_batch_size=2000 --cf_batch_size 8192 --kg_batch_size 8192

python -m main_bprmf --data_name amazon-reviews-23 --data_dir /home/kchauhan/repos/mds-tmu-mrp/datasets --n_epoch 10 --test_batch_size=150 --use_pretrain 0

 python -m main_bprmf --data_name baseline-kg --data_dir /home/kchauhan/repos/mds-tmu-mrp/datasets --n_epoch 1 --train_batch_size=32768  --test_batch_size=100 --use_pretrain 0

python main_kgat.py 


HAS_SIMILAR_ELEMENTS
DEVELOPED_BY
PUBLISHED_BY
HAS_CHARACTERISTICS
COMPARES_TO
HAS_FEATURE


Label atleast 200
llama - 3.1 8b 
annotate 
Try few shot learning



## Project
- http://localhost:7474/browser/

## MRP
- https://www.torontomu.ca/graduate/datascience/students/MRPguidelinesforms/
- https://www.torontomu.ca/graduate/datascience/students/MRPs/
- https://www.torontomu.ca/content/dam/early-childhood-studies/pdfs/student-resources/Graduate/mrp-guidelines.pdf
- https://www.torontomu.ca/graduate/datascience/faculty/#!accordion-1499783917273-faculty-of-engineering-and-architectural-science

## Datasets
- https://github.com/hyp1231/AmazonReviews2023
- https://amazon-reviews-2023.github.io/
- https://github.com/RUCAIBox/RecSysDatasets
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- https://cseweb.ucsd.edu/~jmcauley/datasets.html

## EDA
- https://www.kaggle.com/code/mohamedbakrey/eda-for-amazon-product-review-sentiment-analysis
- https://github.com/huzaifakhan04/exploratory-data-analysis-on-amazon-review-data-using-mongodb-and-pyspark
- https://www.kaggle.com/code/arhamrumi/amazon-reviews-eda
- https://www.linkedin.com/pulse/amazon-review-sentiment-analysis-report-muhammad-shayan-umar-ukfkf/
- https://github.com/RiyaVachhani/SentimentAnalysis-AmazonReviews/blob/main/Sentiment%20%20Analysis%20on%20Amazon%20Reviews.ipynb
- https://medium.com/@arhamrumi/amazon-reviews-eda-662c485ec00c

## Papers
- https://github.com/WLiK/LLM4Rec-Awesome-Papers
- https://github.com/RUCAIBox/LC-Rec
- https://github.com/yuh-yang/KGCL-SIGIR22
- 
## KG
- https://github.com/HotBento/KG4RecEval
- https://www.reddit.com/r/LocalLLaMA/comments/186qq92/using_mistral_openorca_to_create_a_knowledge/
- https://blog.langchain.dev/constructing-knowledge-graphs-from-text-using-openai-functions/
- https://github.com/fusion-jena/automatic-KG-creation-with-LLM
- Prompt - https://gist.github.com/Tostino/44bbc6a6321df5df23ba5b400a01e37d
- https://towardsdatascience.com/leverage-keybert-hdbscan-and-zephyr-7b-beta-to-build-a-knowledge-graph-33d7534ee01b
- https://www.amazon.science/blog/building-commonsense-knowledge-graphs-to-aid-product-recommendation
- https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
- https://graphacademy.neo4j.com/courses/llm-knowledge-graph-construction/\


## Tools / Libraries
- https://github.com/aws/graph-notebook
- http://nlpprogress.com/english/coreference_resolution.html
- https://aws.amazon.com/neptune/knowledge-graphs-on-aws/

## Other 
- https://github.com/RUCAIBox/LLMRank
- https://python.langchain.com/v0.1/docs/use_cases/graph/semantic/
- 






