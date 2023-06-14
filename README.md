<span style="color:red"> ↑ Note!! Make sure you're on the `main` branch for the latest version of this tutorial. </span>

# Hopfield Network Tutorial

<img src="images/hfn.gif" width=850>

In this tutorial ([hopfield_networks.ipynb](hopfield_networks.ipynb)) - designed for the TReND Comp Neuro and ML summer school 2023 - we will get hands on building Hopfield networks and training them to memorise patterns.  We'll start off simple with a classic Hopfield netowrk which can memorize random binary patterns. Afterwards we'll train an improved version to memorise the 54 African flags. Here's the plan: 

1. Generate some random binary patterns we'll use as our "memories".
2. Make a classic Hopfield Network: calculate the weight martix and define the dynamical update rule.
3. Test it: can the dynamics of our Hopfield network recover the memories we gave it?
4. Improve it: With a simple upgrade to [modern Hopfield Networks](https://ml-jku.github.io/hopfield-layers/#energy) we can remember more complex patterns such as flags.

Flags data is pickled inside `flags_of_africa.pickle`. This will be downloaded automatically within the script. 

We recommend cloning and running on your local IDE (it isn't compute heavy and won't require GPUs), but you can also run remotely on Google colab here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomGeorge1234/HopfieldNetworkTutorial/blob/main/hopfield_networks.ipynb).

This lives as a submodule in the TReNDs teaching repo (test2)