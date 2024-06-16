https://github.com/aws/graph-notebook
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

https://amazon-reviews-2023.github.io


https://github.com/fusion-jena/automatic-KG-creation-with-LLM