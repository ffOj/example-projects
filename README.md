# Project description:
The project comprises two parts.
- The first part is contained in the 'learning' folder. It contains reproductions of my favourite parts of a list of work I enjoyed reading. It mainly contains learning rules such as (direct-) feedback alignment, surrogate gradients for SNNs, actor-critic, etc. as well as network models such as variational autoencoders.
This work is written in python using TensorFlow. This was also the first time learning TensorFlow for me, as I am mainly using PyTorch. 
- Secondly, there is the 'AIlib' folder I wrote as an undergrad. I was interested in how automatic differentiation can work being the underlying machinery for deep learning frameworks. Since I enjoy functional programming a lot, I decided to write it from scratch in scala, which comes with serious performance issues due to execution purely on cpu. However, it shows how scala macros could be used to implement efficient compile time computation tree creation.
