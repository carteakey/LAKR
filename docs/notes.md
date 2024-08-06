

MATCH (a:Book)-[rel:SIMILAR_TO_BOOK]->(a) 
DELETE rel;


```
CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install pandas numpy scikit-learn scipy implicit
```

python -m main_bprmf --data_name amazon-book --data_dir ~/repos/mds-tmu-mrp/datasets --n_epoch 5 --test_batch_size=150 --use_pretrain 0 --train_batch_size=32768  


nohup python -m main_bprmf --data_name baseline-kg --data_dir ~/repos/mds-tmu-mrp/datasets --n_epoch 100 --train_batch_size=32768  --test_batch_size=500 --use_pretrain 0 1> training.log 2> training.err &

nohup python -m main_bprmf --data_name baseline-kg --data_dir ~/mds-tmu-mrp/datasets --n_epoch 100 --train_batch_size=32768 --test_batch_size=500 --use_pretrain 0 1> training.log 2> training.err &

2024-07-31 00:21:03,713 - INFO - Total items processed: 29475453
2024-07-31 00:21:03,713 - INFO - Total items inserted: 9543016
2024-07-31 00:21:03,714 - INFO - Reviews for Books loaded
2024-07-31 00:21:03,714 - INFO - Data insertion complete
2024-07-31 00:21:03,795 - INFO - Table 'rating_only' has 9488297 rows
2024-07-31 00:21:03,796 - INFO - Table 'rating_only_positive' has 8038735 rows
2024-07-31 00:21:03,797 - INFO - Table 'raw_meta_Books' has 494691 rows
2024-07-31 00:21:03,798 - INFO - Table 'raw_review_Books' has 9543016 rows 

python main_kgat.py 

CF Evaluation: Epoch 0010 | Total Time 59.9s | Precision [0.0171, 0.0114], Recall [0.0633, 0.2136], NDCG [0.0954, 0.1573]
CF Evaluation: Epoch 0010 | Total Time 68.8s | Precision [0.0159, 0.0107], Recall [0.0597, 0.2021], NDCG [0.0897, 0.1502]

CF Evaluation: Epoch 0020 | Total Time 61.8s | Precision [0.0178, 0.0119], Recall [0.0656, 0.2221], NDCG [0.0992, 0.1627]
CF Evaluation: Epoch 0020 | Total Time 93.8s | Precision [0.0174, 0.0116], Recall [0.0645, 0.2165], NDCG [0.0970, 0.1594]


CF Evaluation: Epoch 0040 | Total Time 57.2s | Precision [0.0187, 0.0125], Recall [0.0696, 0.2321], NDCG [0.1032, 0.1683]
CF Evaluation: Epoch 0040 | Total Time 64.9s | Precision [0.0181, 0.0121], Recall [0.0666, 0.2240], NDCG [0.0999, 0.1636]



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





higher core filtering