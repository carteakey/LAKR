
- Recommender systems have been widely applied to address the issue of information overload in various internet services, exhibiting promising performance in scenarios such as e-commerce platforms and media recommendations.


- RS methods are mainly categorized into Collaborative Filtering (CF), Content-Based Filtering (CBF), and hybrid recommender system based on the input data.


- In the general domain, the traditional knowledge recommendation method is collaborative filtering (CF), which usually suffers from the cold start problem and sparsity of user-item interactions.


- Graph Convolution Network (GCN) has become one of the new state-of-the-art for collaborative filtering.
	https://paperswithcode.com/sota/recommendation-systems-on-amazon-book
	https://paperswithcode.com/task/recommendation-systems


- To address limitations, incorporating knowledge graphs (KG) as side information to improve the recommendation performance has attracted attention.


- However, KGs are difficult to construct and evolve by nature, and existing methods often lack considering textual information. LLM-augmented KGs, that leverage Large Language models (LLM) for different KG tasks such as embedding, completion, construction can be a way to help overcome these challenges and lead to better recommendation systems.

### Areas of Research

- Can we use LLM's to enhance the construction / quality / volume of information in the knowledge graphs?

![[Idea.png]]

- Constraining LLM output to be of a specific format to extract the entities and relations.
	- Format enforcers ([GBNF](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) grammars, and other similar tools that can constrain output include [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer), [Jsonformer](https://github.com/1rgs/jsonformer), and [Outlines](https://github.com/outlines-dev/outlines).)
	- Fine Tuning ??

- Combining that with existing solutions.

### **Dataset** - https://amazon-reviews-2023.github.io
# Potential Issues

- High cost of inference to construct graphs with LLM.
- LLM hallucinations / enforcing output. 
- BERT and existing NLP solutions may be better.

# Examples

A. T. Wasi, T. H. Rafi, R. Islam, and D.-K. Chae, “BanglaAutoKG: Automatic Bangla Knowledge Graph Construction with Semantic Neural Graph Filtering.” arXiv, Apr. 05, 2024. Accessed: Apr. 08, 2024. [Online]. Available: [http://arxiv.org/abs/2404.03528](http://arxiv.org/abs/2404.03528)

![[AutoKG.png ]]
S. Pan, L. Luo, Y. Wang, C. Chen, J. Wang, and X. Wu, “Unifying Large Language Models and Knowledge Graphs: A Roadmap,” _IEEE Trans. Knowl. Data Eng._, pp. 1–20, 2024, doi: [10.1109/TKDE.2024.3352100](https://doi.org/10.1109/TKDE.2024.3352100).
![[KG_construction.png]]

S. Yang _et al._, “Common Sense Enhanced Knowledge-based Recommendation with Large Language Model.” arXiv, Mar. 27, 2024. Accessed: Mar. 30, 2024. [Online]. Available: [http://arxiv.org/abs/2403.18325](http://arxiv.org/abs/2403.18325)

![[CSRec.png]]



baseline
risky
Add to user-item graph.
