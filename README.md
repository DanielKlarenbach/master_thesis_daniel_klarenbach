# Directory naming
example: min_cut_pool_50_size_500_sup_5_hid_200_emb_1_aggr
- min_cut_pool - name of the pooling module
- 50_size - image size
- 500_sup - declared number of superpixels
- 5_hid - number of total GNN layers minus one, so for this particular experiment 6 layers were used
- 200_emb - size of node embeddings
- 1_aggr - number of aggregations

# Relevant file naming
- u.py - complete pipeline that segments images using the implemented superpixel segmentation algorithm
- visual_result* - results of the segmentation
- r.sh - file used to run the u.py script on the Athena supercomputer
