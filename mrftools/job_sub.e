Traceback (most recent call last):
  File "./script/parallelization_comparer_fan_yao.py", line 47, in <module>
    images, models, labels, names = loader.load_all_images_and_labels(path+'/ysirui/data/horse/train', 2, num_training_images)
  File "/work/hokieone/ysirui/script/ImageLoader.py", line 70, in load_all_images_and_labels
    model = ImageLoader.create_model(img, num_states)
  File "/work/hokieone/ysirui/script/ImageLoader.py", line 90, in create_model
    tree_prob = ImageLoader.calculate_tree_probabilities_snake_shape(img.width, img.height)
  File "/opt/apps/Anaconda/2.3.0/lib/python2.7/site-packages/PIL/Image.py", line 622, in __getattr__
    raise AttributeError(name)
AttributeError: width
