

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
CF Evaluation: Epoch 0010 | Total Time 153.6s | Precision [0.0171, 0.0114], Recall [0.0637, 0.2131], NDCG [0.0953, 0.1571]
CF Evaluation: Epoch 0010 | Total Time 58.0s | Precision [0.0171, 0.0114], Recall [0.0639, 0.2120], NDCG [0.0954, 0.1567]
CF Evaluation: Epoch 0010 | Total Time 60.0s | Precision [0.0172, 0.0114], Recall [0.0634, 0.2126], NDCG [0.0951, 0.1569]
CF Evaluation: Epoch 0010 | Total Time 65.8s | Precision [0.0169, 0.0114], Recall [0.0626, 0.2120], NDCG [0.0946, 0.1564]
CF Evaluation: Epoch 0010 | Total Time 66.9s | Precision [0.0171, 0.0114], Recall [0.0633, 0.2124], NDCG [0.0964, 0.1573]
CF Evaluation: Epoch 0010 | Total Time 59.9s | Precision [0.0171, 0.0114], Recall [0.0633, 0.2136], NDCG [0.0954, 0.1573]
CF Evaluation: Epoch 0010 | Total Time 68.8s | Precision [0.0159, 0.0107], Recall [0.0597, 0.2021], NDCG [0.0897, 0.1502]

CF Evaluation: Epoch 0020 | Total Time 61.8s | Precision [0.0178, 0.0119], Recall [0.0656, 0.2221], NDCG [0.0992, 0.1627]
CF Evaluation: Epoch 0020 | Total Time 93.8s | Precision [0.0174, 0.0116], Recall [0.0645, 0.2165], NDCG [0.0970, 0.1594]

BPRMF
CF Evaluation: Epoch 0040 | Total Time 41.8s | Precision [0.0090, 0.0059], Recall [0.0349, 0.1167], NDCG [0.0516, 0.0896]
KGAT
CF Evaluation: Epoch 0040 | Total Time 64.9s | Precision [0.0181, 0.0121], Recall [0.0666, 0.2240], NDCG [0.0999, 0.1636]
KGAT-LLM
CF Evaluation: Epoch 0040 | Total Time 57.2s | Precision [0.0187, 0.0125], Recall [0.0696, 0.2321], NDCG [0.1032, 0.1683]
CF Evaluation: Epoch 0040 | Total Time 59.6s | Precision [0.0187, 0.0125], Recall [0.0688, 0.2316], NDCG [0.1035, 0.1686]


Epoch 0040 | Total Time 66.2s | Precision [0.0186, 0.0125], Recall [0.0691, 0.2321], NDCG [0.1037, 0.1688]

Epoch 0090 | Precision [0.0197, 0.0131], Recall [0.0727, 0.2430], NDCG [0.1090, 0.1755]

Epoch 0090 | Precision [0.0193, 0.0127], Recall [0.0712, 0.2359], NDCG [0.1065, 0.1712]
Epoch 0090 | Precision [0.0199, 0.0130], Recall [0.0737, 0.2414], NDCG [0.1100, 0.1756]
Epoch 0090 | Precision [0.0198, 0.0130], Recall [0.0735, 0.2417], NDCG [0.1084, 0.1751]s 

Epoch 0100 | Total Time 70.1s | Precision [0.0196, 0.0131], Recall [0.0722, 0.2433], NDCG [0.1076, 0.1747]

Validation Metrics of Best Model:
{'precision@10': 0.0815, 'MAP@10': 0.0342, 'NDCG@10': 0.0586}

Test Metrics of Best Model:
{'precision@10': 0.0879, 'MAP@10': 0.0375, 'NDCG@10': 0.0772}


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

find . -type f -name "*.log" -size -1k -delete
find . -type f -name "*.log" -size -10 -delete


python -m predict_kgat --data_name baseline-kg --data_dir ~/repos/mds-tmu-mrp/data/kg --pretrain_model_path /home/kchauhan/repos/mds-tmu-mrp/src/trained_model/KGAT/baseline-kg/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain1/model_epoch90.pth --pretrain_embedding_dir /home/kchauhan/repos/mds-tmu-mrp/data/kg/baseline-kg/pretrain