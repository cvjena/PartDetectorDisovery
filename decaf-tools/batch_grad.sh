source ../python_virtual_env/python-env/bin/activate
python extract_grad_map.py --images $(cat /home/simon/Datasets/CU_Dogs/imagelist_small.txt) --limit 1 --channel_limit 256 --layers probs pool5 --outdir /home/simon/tmp/dog-grads/
