# ua-dep-parser
A project for the ML course at UCU DS Master's program. It implements an ensemble model for the Ukrainian dependency parsing.

[The corpus of dependencies in Ukrainian](https://github.com/UniversalDependencies/UD_Ukrainian-IU) is taken from the official universal dependencies github (in `/data`).
 

**How the ensemble works:**

Initially, it has been implemented on the base of a general hard voting algorithm. For each tree node, it gets the predictions from N models and merges those predictions properities into single node based on values frequencies. As the result, it outputs a tree of merged nodes.

However, sometimes it happens that after merging, the tree may contain circular dependencies. Therefore, the soft voting has been added additionally to handle this issue.  Basically, it calculates match ratios between the models and selects the prediction from the single model that has the most similaries compared to other models.

How to run the project:

- Install all required packages

- Run `python app.py`

The application contains data loader implementation, so all required data will be downloaded automatically.

Happy usage!
