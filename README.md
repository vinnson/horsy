# This is available to run under tensorflow/tensorflow container image
# Sample Command to run retrain
$ python retrain.py --image_dir /opt/tensor/images/ --model_dir model/ --output_graph output_graph.pb --output_labels output_labels.txt

# Sample Command to label image
python label_image.py --image /opt/images/test.JPG --graph output_graph.pb --labels output_labels.txt
