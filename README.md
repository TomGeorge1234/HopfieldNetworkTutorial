# Hopfield Network Tutorial

In this tutorial ([hopfield_networks.ipynb](hopfield_networks.ipynb)) - designed for the TReND Comp Neuro and ML summer school 2023 - we will get hands on building Hopfield networks and training them to memorise patterns.  We'll start off simple with a classic Hopfield netowrk which can memorize random binary patterns. Afterwards we'll train an improved version to memorise the 54 African flags. Here's the plan: 

1. Generate some random binary patterns we'll use as our "memories" 
2. Make a classic Hopfield Network: calcualte the weight martix and define the dynamical update rule
3. Test it
4. Improve it: With a simple upgrade to modern Hopfield Networks we can remember more complex patterns such as flags

Linear regression: A simple model with an analytic solution. We'll use this as a comparison later on.

We recommend cloning and running on your local IDE (it isn't compute heavy and won't require GPUs), but you can also run remotely on Google colab here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomGeorge1234/HopfieldNetworkTutorial/blob/main/hopfield_networks.ipynb).