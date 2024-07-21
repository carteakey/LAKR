## Tasks

- [ ] 5 Sample Records
- [ ] 

## Project
- http://localhost:7474/browser/

**DS8013**
Office Hours
[https://torontomu.zoom.us/j/98824233888?pwd=MEdKWmRLQTJtTUIvbGpmQnhFTyszUT09](https://torontomu.zoom.us/j/98824233888?pwd=MEdKWmRLQTJtTUIvbGpmQnhFTyszUT09)  

**DS8004**
Office Hours
[https://torontomu.zoom.us/j/98824233888?pwd=MEdKWmRLQTJtTUIvbGpmQnhFTyszUT09](https://torontomu.zoom.us/j/98824233888?pwd=MEdKWmRLQTJtTUIvbGpmQnhFTyszUT09)  
  
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




docker run \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/neo4j/data:/data \
    neo4j
    --env='NEO4JLABS_PLUGINS=["apoc"]'



```
CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install pandas numpy scikit-learn scipy implicit
```





