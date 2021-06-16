# ua-dep-parser
A project for the ML course at UCU DS Master's program. It implements an ensemble model for the Ukrainian dependency parsing.

[The corpus of dependencies in Ukrainian](https://github.com/UniversalDependencies/UD_Ukrainian-IU) is taken from the official universal dependencies github (in `/data`).
 
**What is dependency parsing:**

Dependency parsing is a representation of a sentence in a form of a labelled binary directed graph where arcs between two words stand for grammatical relations. These grammatical relations are used as features in many NLP tasks (e.g. relations extraction) – dependency parsing is a downstream task.

In this project, we were trying to improve existing dependency parsers for Ukrainian by creating an ensemble model of several parsers.


**How the ensemble works:**

Initially, it has been implemented on the base of a general hard voting algorithm. For each tree node, it gets the predictions from N models and merges those predictions properities into single node based on values frequencies. As the result, it outputs a tree of merged nodes.

However, sometimes it happens that after merging, the tree may contain circular dependencies. Therefore, the soft voting has been added additionally to handle this issue.  Basically, it calculates match ratios between the models and selects the prediction from the single model that has the most similaries compared to other models.

**How to run the project:**

- Clone this repository to your local machine. Go to the root of the repostiory in you terminal.

- Create a virtual environment `python -m venv ua-ensemble-parser`

- Switch to virtual environment  `source ua-ensemble-parser/bin/activate`

- Install all required packages  `pip install -r requirements.txt`

- Run `python app.py`. The application contains data loader implementation, so all required data will be downloaded automatically. This may take a while, so be patient 😉. 

- When the vectors and models are downloaded, open `http://192.168.0.101:5000/` and try parsing any Ukrainian sentence. You should get a graph in a json format.

