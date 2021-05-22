# ua-dep-parser
A project for the ML course at UCU DS Master's program.

Aim - increase the accuracy of the Ukrainian dependency parser from [stanza](https://github.com/stanfordnlp/stanza).

[The corpus of dependencies in Ukrainian](https://github.com/UniversalDependencies/UD_Ukrainian-IU) is taken from the official universal dependencies github (in `/data`).

 

How ensemble works:

Initially, it has been implemented on the base of general hard voting algorithm. For each tree node, it gets the predictions from N models and merges those predictions properities into single node based on values frequencies. As the result, it outputs the tree of merged nodes.
Unlikely, it has been noticed sometimes it happens that after merging, the nodes tree may contain circular dependencies. 
Therefore, the soft voting has been added additionally to handle this issue.  Basically, it calculates the matches ratios between the models and select the prediction from the single model that has the most similaries comparing to other models results.

How to run project:
Intall all required packages and run

python app.py

The application contains data loader implementation, so all required data will be downloaded automatically.

Happy usage!