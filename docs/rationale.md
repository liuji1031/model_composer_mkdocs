# Why this could be useful

## Problem
***Have you ever wanted to understand the architecture of a neural network, say from a repository accompanying a research paper, but found it difficult to dig through the code?*** 

I have experienced such frustration many times. The paper might not provide enough details, and the code might be too complex to follow, especially when it is structured to allow for flexibility and modularity for training and experimentation. The issue is exacerbated by the following: <br/>

1. The modules that constitute the architecture might be scattered across multiple files, and the main file might not provide a clear overview of how the modules are <span style="color:#1f57ff">***connected***</span>. 
2. The build configuration files are usually <span style="color:#1f57ff">***separate***</span> from the model definition files, making it difficult to understand how the model is built and what parameters are used.
3. The repository might also contain many <span style="color:#1f57ff">***different versions***</span> of the model that the author experimented with, making it even more confusing to understand the final architecture.

Therefore, I wanted to create an intuitive format to describe the architecture of a neural network that will make it easier to convey the core ideas to the readers. I hope to achieve the following goals: <br/>

1. The format should be <span style="color:#ff6a00">***easy to read and write***</span>.
2. It should contain not only module configurations (hyperparameter), but also the <span style="color:#ff6a00">***connections***</span> between the modules.
3. It should allow the user to choose a level of <span style="color:#ff6a00">***granularity***</span> best for conveying the core ideas.
4. It can be readily parsed by a Python script to <span style="color:#ff6a00">***build***</span> the model.
5. It can be easily parsed by a visualization tool to create a <span style="color:#ff6a00">***diagram***</span> of the architecture.

## Solution
To this end, I designed the YAML format for ***ComposableModel*** class and built a visualization tool ([***Neural Network Architecture Visualizer***](https://network-visualizer-36300.web.app/)) to achieve the goals mentioned above. 

The key benefits of this approach are:

1. The final architecture can be described in a single or a suite of YAML files. Housing them with specific directory structures makes it easy to understand the architecture at a glance, without having to dig through the code.
2. The YAML files include how the connections between the modules are made, alongside the build configuration for each module. 
3. It is easy to swap out different modules in the architecture without having to change the code. This is especially useful for research purposes, where the user might want to experiment with different modules or configurations.
4. The YAML files can be written in a modular way, allowing the user to define the architecture at any level of granularity. 
5. The visualization tool creates a high-quality diagram of the architecture from the YAML file, which can be readily used in a paper or a presentation.
6. The YAML format can be used to create abstract or conceptual architectures (detached from model building process), which can be useful for quickly sketching out ideas or for creating diagrams for papers or presentations.

A typical pipeline for using the YAML format can look like this:

1. The user writes custom modules in Python as usual through subclassing, etc.
2. The user writes a YAML file that describes the architecture of the model, including the modules and their connections.
3. The user uses the ***ComposableModel*** class to build the final model from the YAML file.
4. The user uses the visualization tool to create a diagram of the architecture from the YAML file to inspect the architecture or use it in a paper or a presentation.

I hope this approach will make it easier for researchers and practitioners to understand and communicate the architecture of neural networks, and to experiment with different architectures in a more intuitive way.

Happy composing!