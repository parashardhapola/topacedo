# TopACeDo
#### Topology Assisted Cell Downsampling

Topacedo works with Scarf to perform memory efficient downsampling of single-cell genomics data.
Using a pre-built neighbourhood graph of cells, Topacedo aims to sample cells from the graph such 
that the manifold is preserved. Topecedo's algorithm can be diveded into two phases:

- assigning prizes to cells and penalties to edges
- running a fast approximate prize-collecting steiner tree  
